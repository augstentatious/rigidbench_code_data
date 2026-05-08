#!/usr/bin/env python3
"""RigidBench v2 — Full Experiment Suite.

Runs all 20 triples × 3 pressure levels against available models,
scores each completion for identity preservation vs semantic vs
phonological substitution, and writes structured results.

Usage:
    # With API keys set in environment:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."

    python run_all.py --models claude-sonnet-4-20250514 gpt-4o --output results/
    python run_all.py --models claude-sonnet-4-20250514 --output results/ --dry-run  # preview prompts
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Triple:
    id: str
    proper_noun: str
    semantic_lure: str
    phonological_neighbor: str
    phon_distance_name_to_lure: int
    phon_distance_name_to_neighbor: int
    semantic_sim_name_to_lure: float
    semantic_sim_name_to_neighbor: float
    etymological_link: bool
    pressure_low: str
    pressure_mid: str
    pressure_high: str
    expected_output: str
    notes: str = ""


@dataclass
class Result:
    triple_id: str
    model: str
    pressure_level: str  # "low", "mid", "high", or "baseline"
    prompt: str
    proper_noun: str
    semantic_lure: str
    phonological_neighbor: str
    raw_completion: str
    matched_word: str  # was "first_word"; now stores the word that matched
    error_type: str  # "preserved", "semantic_sub", "phonological_sub", "other"
    phon_distance_name_to_lure: int
    phon_distance_name_to_neighbor: int
    semantic_sim_name_to_lure: float
    semantic_sim_name_to_neighbor: float
    etymological_link: bool
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------


def query_anthropic(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def query_openai(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query OpenAI API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _min_tokens_for_reasoning_model(model: str, max_tokens: int) -> int:
    """Some hosted reasoning models spend small token budgets before final text."""
    lower = model.lower()
    if any(
        name in lower
        for name in ("deepseek-v4", "gpt-oss", "kimi-k2", "gemini-2.5", "gemini-3")
    ) or any(name in lower for name in ("grok-4", "grok-4.3", "gpt-5.4", "gpt-5.5")):
        floor = int(os.environ.get("REASONING_MODEL_MIN_OUTPUT_TOKENS", "1024"))
        return max(max_tokens, floor)
    return max_tokens


def _chat_message_text(resp: Any) -> str:
    """Extract final assistant text from an OpenAI-compatible chat response."""
    choice = resp.choices[0]
    content = getattr(choice.message, "content", None)
    if content is None:
        finish_reason = getattr(choice, "finish_reason", None)
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
            choice.message, "reasoning", None
        )
        reasoning_len = len(reasoning) if reasoning else 0
        raise RuntimeError(
            "Model returned empty final content "
            f"(finish_reason={finish_reason}, reasoning_len={reasoning_len})"
        )
    return content.strip()


def _chat_stream_text(stream: Any) -> str:
    """Extract assistant text from an OpenAI-compatible streaming response."""
    chunks: list[str] = []
    finish_reason = None
    reasoning_len = 0
    for event in stream:
        if not getattr(event, "choices", None):
            continue
        choice = event.choices[0]
        finish_reason = getattr(choice, "finish_reason", finish_reason)
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            chunks.append(content)
        reasoning = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reasoning:
            reasoning_len += len(reasoning)

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError(
            "Model returned empty final content "
            f"(finish_reason={finish_reason}, reasoning_len={reasoning_len})"
        )
    return text


def _configure_vertex_adc() -> None:
    """Prefer an explicit ADC path, then local Windows/WSL gcloud ADC files."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    account = os.environ.get("VERTEX_ADC_ACCOUNT", "cleanroomresearch@gmail.com")
    candidates: list[pathlib.Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(
            pathlib.Path(appdata)
            / "gcloud"
            / "legacy_credentials"
            / account
            / "adc.json"
        )
    candidates.append(
        pathlib.Path.home()
        / ".config"
        / "gcloud"
        / "legacy_credentials"
        / account
        / "adc.json"
    )

    for path in candidates:
        if path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
            return


def _vercel_extra_body() -> dict[str, Any] | None:
    """Optional Vercel AI Gateway reasoning controls via environment variables."""
    extra: dict[str, Any] = {}
    effort = os.environ.get("VERCEL_REASONING_EFFORT")
    if effort:
        extra["reasoning"] = {"effort": effort}

    include_reasoning = os.environ.get("VERCEL_INCLUDE_REASONING")
    if include_reasoning is not None:
        extra["include_reasoning"] = include_reasoning.lower() in (
            "1",
            "true",
            "yes",
        )

    return extra or None


def query_googleai(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Gemini through the Google AI API key backend."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("pip install google-generativeai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable")

    if model.startswith("googleai/"):
        model = model[len("googleai/") :]
    if model.startswith("models/"):
        model = model[len("models/") :]

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=_min_tokens_for_reasoning_model(model, max_tokens),
        ),
    )
    return response.text.strip()


def query_gemini(
    model: str, prompt: str, max_tokens: int = 20, thinking: str = None
) -> str:
    """Query Google Gemini API via Vertex AI SDK."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, GenerationConfig
    except ImportError:
        raise RuntimeError("pip install google-cloud-aiplatform")

    _configure_vertex_adc()

    import os
    from google.oauth2.credentials import Credentials

    token = os.environ.get("VERTEX_ACCESS_TOKEN")
    creds = Credentials(token) if token else None
    vertexai.init(
        credentials=creds,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "paper-493620"),
        location=os.environ.get("VERTEX_LOCATION", "us-central1"),
    )

    if model.startswith("vertex/"):
        model = model[len("vertex/") :]

    gen_model = GenerativeModel(model)

    thinking_config = None
    if thinking and thinking != "off":
        if thinking.startswith("budget_tokens="):
            budget = int(thinking.split("=")[1])
            thinking_config = {"thinking_budget": budget}
        else:
            thinking_config = {"thinking_budget": 1024}

    try:
        response = gen_model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                thinking_config=thinking_config,
                temperature=0.0,
                max_output_tokens=_min_tokens_for_reasoning_model(model, max_tokens),
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def query_anthropic_vertex(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Anthropic API via Google Cloud Vertex AI."""
    try:
        from anthropic import AnthropicVertex
    except ImportError:
        raise RuntimeError("pip install 'anthropic[vertex]'")

    if model.startswith("vertex/"):
        model = model[len("vertex/") :]

    # Initialize client (will use Application Default Credentials)
    client = AnthropicVertex(
        project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "paper-493620"),
        region=os.environ.get("ANTHROPIC_VERTEX_REGION", "global"),
    )

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def query_openrouter(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query OpenRouter API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if model.startswith("openrouter/"):
        model = model[len("openrouter/") :]

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/anonymous/rigidbench",
            "X-Title": "RigidBench",
        },
    )
    return resp.choices[0].message.content.strip()


def query_groq(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Groq API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    if model.startswith("groq/"):
        model = model[len("groq/") :]

    token_budget = _min_tokens_for_reasoning_model(model, max_tokens)
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_budget,
        "temperature": 0.0,
    }
    if token_budget > 4096:
        kwargs["stream"] = True
        return _chat_stream_text(client.chat.completions.create(**kwargs))

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_fireworks(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Fireworks API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Set FIREWORKS_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )

    if model.startswith("fireworks/"):
        model = model[len("fireworks/") :]

    token_budget = _min_tokens_for_reasoning_model(model, max_tokens)
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": token_budget,
        "temperature": 0.0,
    }
    if token_budget > 4096:
        kwargs["stream"] = True
        return _chat_stream_text(client.chat.completions.create(**kwargs))

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_vercel(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query Vercel AI Gateway via its OpenAI-compatible endpoint."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get(
        "VERCEL_OIDC_TOKEN"
    )
    if not api_key:
        raise RuntimeError(
            "Set AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN environment variable"
        )

    client = openai.OpenAI(
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=api_key,
        timeout=float(os.environ.get("VERCEL_REQUEST_TIMEOUT", "120")),
    )

    if model.startswith("vercel/"):
        model = model[len("vercel/") :]

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": _min_tokens_for_reasoning_model(model, max_tokens),
        "temperature": 0.0,
    }
    extra_body = _vercel_extra_body()
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_bedrock(model: str, prompt: str, max_tokens: int = 20) -> str:
    """Query AWS Bedrock API."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("pip install boto3")

    if model.startswith("bedrock/"):
        model = model[len("bedrock/") :]

    client = boto3.client("bedrock-runtime")

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    response = client.converse(
        modelId=model,
        messages=messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.0,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def query_model(model: str, prompt: str) -> str:
    """Route to the correct backend."""
    if model.startswith("openrouter/") or ("/" in model and "openrouter" in model):
        return query_openrouter(model, prompt)
    elif model.startswith("googleai/"):
        return query_googleai(model, prompt)
    elif model.startswith("groq/"):
        return query_groq(model, prompt)
    elif model.startswith("fireworks/"):
        return query_fireworks(model, prompt)
    elif model.startswith("vercel/"):
        return query_vercel(model, prompt)
    elif model.startswith("bedrock/"):
        return query_bedrock(model, prompt)
    elif (
        "claude" in model.lower()
        or "sonnet" in model.lower()
        or "opus" in model.lower()
        or "haiku" in model.lower()
    ):
        # If it's a Vertex AI model string (e.g., claude-3-5-sonnet@20240620 or anthropic-claude-...), route to Vertex
        if (
            "@" in model
            or "vertex" in model.lower()
            or model.startswith("anthropic-claude")
        ):
            return query_anthropic_vertex(model, prompt)
        else:
            return query_anthropic(model, prompt)
    elif (
        "gpt" in model.lower()
        or "o1" in model.lower()
        or "o3" in model.lower()
        or "o4" in model.lower()
    ):
        return query_openai(model, prompt)
    elif "gemini" in model.lower():
        return query_gemini(model, prompt)
    else:
        # Default to OpenAI-compatible
        return query_openai(model, prompt)


# ---------------------------------------------------------------------------
# Multi-turn model backends (v3.1)
# ---------------------------------------------------------------------------


def query_anthropic_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    """Query Anthropic API with multi-turn messages."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("pip install anthropic")

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=messages,
    )
    return resp.content[0].text.strip()


def query_openai_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    """Query OpenAI API with multi-turn messages."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def query_googleai_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    """Query Gemini through Google AI API with chat-style contents."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("pip install google-generativeai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable")

    if model.startswith("googleai/"):
        model = model[len("googleai/") :]
    if model.startswith("models/"):
        model = model[len("models/") :]

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [msg["content"]]})

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(
        contents,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=_min_tokens_for_reasoning_model(model, max_tokens),
        ),
    )
    return response.text.strip()


def query_gemini_multiturn(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 20,
    thinking: str = None,
) -> str:
    """Query Google Gemini API via Vertex AI SDK with multi-turn messages."""
    try:
        import vertexai
        from vertexai.generative_models import (
            GenerativeModel,
            GenerationConfig,
            Content,
            Part,
        )
    except ImportError:
        raise RuntimeError("pip install google-cloud-aiplatform")

    _configure_vertex_adc()

    import os
    from google.oauth2.credentials import Credentials

    token = os.environ.get("VERTEX_ACCESS_TOKEN")
    creds = Credentials(token) if token else None
    vertexai.init(
        credentials=creds,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "paper-493620"),
        location=os.environ.get("VERTEX_LOCATION", "us-central1"),
    )

    if model.startswith("vertex/"):
        model = model[len("vertex/") :]

    gen_model = GenerativeModel(model)

    thinking_config = None
    if thinking and thinking != "off":
        if thinking.startswith("budget_tokens="):
            budget = int(thinking.split("=")[1])
            thinking_config = {"thinking_budget": budget}
        else:
            thinking_config = {"thinking_budget": 1024}

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(Content(role=role, parts=[Part.from_text(msg["content"])]))

    try:
        response = gen_model.generate_content(
            contents,
            generation_config=GenerationConfig(
                thinking_config=thinking_config,
                temperature=0.0,
                max_output_tokens=_min_tokens_for_reasoning_model(model, max_tokens),
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def query_anthropic_vertex_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    """Query Anthropic API via Vertex AI with multi-turn messages."""
    try:
        from anthropic import AnthropicVertex
    except ImportError:
        raise RuntimeError("pip install 'anthropic[vertex]'")

    if model.startswith("vertex/"):
        model = model[len("vertex/") :]

    client = AnthropicVertex(
        project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "paper-493620"),
        region=os.environ.get("ANTHROPIC_VERTEX_REGION", "global"),
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=messages,
    )
    return response.content[0].text.strip()


def query_openrouter_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if model.startswith("openrouter/"):
        model = model[len("openrouter/") :]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_headers={
            "HTTP-Referer": "https://github.com/anonymous/rigidbench",
            "X-Title": "RigidBench",
        },
    )
    return resp.choices[0].message.content.strip()


def query_groq_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    if model.startswith("groq/"):
        model = model[len("groq/") :]

    token_budget = _min_tokens_for_reasoning_model(model, max_tokens)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": token_budget,
        "temperature": 0.0,
    }
    if token_budget > 4096:
        kwargs["stream"] = True
        return _chat_stream_text(client.chat.completions.create(**kwargs))

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_fireworks_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Set FIREWORKS_API_KEY environment variable")

    client = openai.OpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )

    if model.startswith("fireworks/"):
        model = model[len("fireworks/") :]

    token_budget = _min_tokens_for_reasoning_model(model, max_tokens)
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": token_budget,
        "temperature": 0.0,
    }
    if token_budget > 4096:
        kwargs["stream"] = True
        return _chat_stream_text(client.chat.completions.create(**kwargs))

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_vercel_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("pip install openai")

    api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get(
        "VERCEL_OIDC_TOKEN"
    )
    if not api_key:
        raise RuntimeError(
            "Set AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN environment variable"
        )

    client = openai.OpenAI(
        base_url="https://ai-gateway.vercel.sh/v1",
        api_key=api_key,
        timeout=float(os.environ.get("VERCEL_REQUEST_TIMEOUT", "120")),
    )

    if model.startswith("vercel/"):
        model = model[len("vercel/") :]

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": _min_tokens_for_reasoning_model(model, max_tokens),
        "temperature": 0.0,
    }
    extra_body = _vercel_extra_body()
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    return _chat_message_text(resp)


def query_bedrock_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    try:
        import boto3
    except ImportError:
        raise RuntimeError("pip install boto3")

    if model.startswith("bedrock/"):
        model = model[len("bedrock/") :]

    client = boto3.client("bedrock-runtime")

    bedrock_messages = []
    for msg in messages:
        bedrock_messages.append(
            {"role": msg["role"], "content": [{"text": msg["content"]}]}
        )

    response = client.converse(
        modelId=model,
        messages=bedrock_messages,
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0.0,
        },
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def query_model_multiturn(
    model: str, messages: list[dict[str, str]], max_tokens: int = 20
) -> str:
    """Route multi-turn messages to the correct backend."""
    if model.startswith("openrouter/") or ("/" in model and "openrouter" in model):
        return query_openrouter_multiturn(model, messages, max_tokens)
    elif model.startswith("googleai/"):
        return query_googleai_multiturn(model, messages, max_tokens)
    elif model.startswith("groq/"):
        return query_groq_multiturn(model, messages, max_tokens)
    elif model.startswith("fireworks/"):
        return query_fireworks_multiturn(model, messages, max_tokens)
    elif model.startswith("vercel/"):
        return query_vercel_multiturn(model, messages, max_tokens)
    elif model.startswith("bedrock/"):
        return query_bedrock_multiturn(model, messages, max_tokens)
    elif (
        "claude" in model.lower()
        or "sonnet" in model.lower()
        or "opus" in model.lower()
        or "haiku" in model.lower()
    ):
        if (
            "@" in model
            or "vertex" in model.lower()
            or model.startswith("anthropic-claude")
        ):
            return query_anthropic_vertex_multiturn(model, messages, max_tokens)
        else:
            return query_anthropic_multiturn(model, messages, max_tokens)
    elif (
        "gpt" in model.lower()
        or "o1" in model.lower()
        or "o3" in model.lower()
        or "o4" in model.lower()
    ):
        return query_openai_multiturn(model, messages, max_tokens)
    elif "gemini" in model.lower():
        return query_gemini_multiturn(model, messages, max_tokens)
    else:
        # Default to OpenAI-compatible
        return query_openai_multiturn(model, messages, max_tokens)


# ---------------------------------------------------------------------------
# Scoring — robust regex-based classification
# ---------------------------------------------------------------------------


def classify_error(
    completion: str,
    proper_noun: str,
    semantic_lure: str,
    phonological_neighbor: str,
) -> tuple[str, str]:
    """Classify the completion by scanning the FULL text for word-boundary
    matches of target / lure / neighbor.  Returns (matched_word, error_type).

    Handles both single-word and multi-word names.  For multi-word names
    (e.g. "Grace Kelly"), first tries the full phrase, then falls back to
    checking each individual word.  This prevents misses when a model outputs
    only a last name or first name of a multi-word entity.
    """
    comp_lower = completion.lower()

    def find_phrase(phrase: str) -> int:
        """Return position of phrase (or any individual word in it) in
        completion, or -1 if none found.  Prefers full-phrase match."""
        if not phrase:
            return -1
        # Try full phrase first
        escaped = re.escape(phrase.lower())
        m = re.search(r"\b" + escaped + r"\b", comp_lower)
        if m:
            return m.start()
        # Fallback: try each word of a multi-word name
        words = phrase.split()
        if len(words) > 1:
            for word in words:
                m = re.search(r"\b" + re.escape(word.lower()) + r"\b", comp_lower)
                if m:
                    return m.start()
        return -1

    pos_target = find_phrase(proper_noun)
    pos_lure = find_phrase(semantic_lure)
    pos_neighbor = find_phrase(phonological_neighbor)

    # Collect all matches with their positions
    matches = []
    if pos_target >= 0:
        matches.append((pos_target, "preserved", proper_noun))
    if pos_lure >= 0:
        matches.append((pos_lure, "semantic_sub", semantic_lure))
    if pos_neighbor >= 0:
        matches.append((pos_neighbor, "phonological_sub", phonological_neighbor))

    if not matches:
        # Fallback: return first token for logging
        parts = completion.split()
        first = parts[0] if parts else ""
        return first, "other"

    # If multiple names appear, take the FIRST one in the text
    matches.sort(key=lambda x: x[0])
    _, error_type, matched_word = matches[0]
    return matched_word, error_type


# ---------------------------------------------------------------------------
# v3.1 outcome classification
# ---------------------------------------------------------------------------

# Regex patterns that indicate a model is asking for clarification or abstaining
_CLARIFY_PATTERNS = [
    r"which\s+(one|person|cousin|sister|brother|daughter|nephew|niece|friend|colleague|cat|dog|pet|anchor|kayaker)",
    r"(could you|can you)\s+(clarify|specify|tell me)",
    r"(not enough|insufficient)\s+information",
    r"(either|both)\s+(could|might|of them)",
    r"(ambiguous|unclear|impossible to (determine|tell|know))",
    r"(I\s+)?can('?t| ?not)\s+(determine|tell|know|be sure)",
    r"(I\s+)?(need|would need)\s+(more|additional)\s+(information|context|details)",
    r"it('s| is)\s+(not clear|unclear|impossible|ambiguous)",
    r"(there is|there's)\s+(no|not enough|insufficient)\s+(way|information)",
    r"(I('m| am)\s+)?(not sure|uncertain|unsure)",
]
_CLARIFY_RE = re.compile("|".join(_CLARIFY_PATTERNS), re.IGNORECASE)

_ABSTAIN_PATTERNS = [
    r"(I\s+)?cannot\s+(confidently|definitively)\s+(say|answer|determine)",
    r"(I\s+)?(don't|do not)\s+have\s+enough",
    r"(I\s+)?would\s+not\s+want\s+to\s+guess",
    r"(I('m| am)\s+)?unable\s+to\s+(determine|answer)",
    r"(this|the)\s+(question|description)\s+(is|seems)\s+(ambiguous|unclear)",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def detect_clarify_or_abstain(completion: str) -> str | None:
    """Check if a completion is a clarification request or abstention.

    Returns 'CLARIFY', 'ABSTAIN', or None.
    """
    if _CLARIFY_RE.search(completion):
        return "CLARIFY"
    if _ABSTAIN_RE.search(completion):
        return "ABSTAIN"
    # Check for question marks in short responses (likely clarification)
    if "?" in completion and len(completion.split()) < 40:
        return "CLARIFY"
    return None


def classify_v3_outcome(
    completion: str,
    item: dict[str, Any],
) -> tuple[str, str]:
    """Classify a v3.1 completion into (matched_word, outcome_code).

    Uses the v3.1 outcome taxonomy: PRES, SEM_SUB, PHO_SUB, ALIAS_OK,
    CLARIFY, ABSTAIN, ENT_CONF, META, NOISE.

    Delegates to the existing classify_error for core name matching,
    then layers on Family-specific logic.
    """
    family = item.get("family", "")
    mention = item.get("mention", "")
    semantic_lure = item.get("semantic_lure", "")
    phon_neighbor = item.get("phonological_neighbor", "N/A")
    valid_aliases = item.get("valid_aliases", [])
    expected_correct = item.get("expected_correct", [])

    # --- Check for clarification / abstention first ---
    ca = detect_clarify_or_abstain(completion)
    if ca is not None:
        return ("", ca)

    # --- Check valid aliases ---
    comp_lower = completion.lower()
    for alias in valid_aliases:
        if alias and re.search(r"\b" + re.escape(alias.lower()) + r"\b", comp_lower):
            return (alias, "ALIAS_OK")

    # --- Core name match via existing classifier ---
    if phon_neighbor == "N/A":
        phon_neighbor = ""
    matched_word, base_error = classify_error(
        completion, mention, semantic_lure, phon_neighbor
    )

    # Map base_error to v3.1 outcome codes
    outcome_map = {
        "preserved": "PRES",
        "semantic_sub": "SEM_SUB",
        "phonological_sub": "PHO_SUB",
        "other": "NOISE",
    }
    outcome = outcome_map.get(base_error, "NOISE")

    # --- Family E: check entity-set confusion ---
    # If the item has multiple entities in the prompt and the output is "other",
    # check if the completion contains any entity name from the prompt set
    if family == "entity_set_competition" and outcome == "NOISE":
        # Extract all capitalized names from the first turn
        first_turn = item.get("prompt_turns", [""])[0]
        # Simple heuristic: look for names introduced with dashes or commas
        candidate_names = re.findall(r"\b([A-Z][a-z]{2,})\b", first_turn)
        for name in candidate_names:
            if name.lower() == mention.lower():
                continue
            if name.lower() == semantic_lure.lower():
                continue
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", comp_lower):
                return (name, "ENT_CONF")

    return (matched_word, outcome)


def build_v3_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages from a v3.1 item's prompt_turns.

    For single-turn items (Family A/C/D/E): all turns concatenated into
    one user message.

    For multi-turn items (Family B): alternating user/assistant turns.
    Each user turn is sent as a user message; the assistant acknowledges
    with a brief continuation to keep the conversation going, except
    the final turn which is the query.
    """
    turns = item.get("prompt_turns", [])
    family = item.get("family", "")

    if family == "multi_turn_persistence" and len(turns) > 1:
        # Multi-turn: build alternating user/assistant messages
        messages: list[dict[str, str]] = []
        for i, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn})
            # After each turn except the last, add a brief assistant ack
            if i < len(turns) - 1:
                messages.append(
                    {"role": "assistant", "content": "I understand. Please continue."}
                )
        return messages
    else:
        # Single-turn: join all turns into one prompt
        combined = "\n".join(turns)
        return [{"role": "user", "content": combined}]


def get_v3_max_tokens(family: str) -> int:
    """Return appropriate max_tokens for a v3.1 family."""
    if family == "summary_compression":
        return 200  # summaries need room
    elif family in ("clarify_abstain", "entity_set_competition"):
        return 100  # may need to explain ambiguity
    else:
        return 30  # completion / persistence: short answer


# ---------------------------------------------------------------------------
# Baseline frequency probe — controls for unigram name frequency (hostile
# review flaw #3).  For each triple, we query the model with a neutral
# context that gives no semantic pressure and see which name it defaults to.
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = "Complete the sentence with a name.\nMy {relation}'s name is"

# Map triple IDs to the neutral relation used in each prompt
_RELATION_BY_ID: dict[str, str] = {}  # populated lazily from triple data


def make_baseline_prompt(triple: Triple) -> str:
    """Build a neutral prompt with zero semantic pressure for this triple.

    Uses a generic relation word derived from the triple's low-pressure prompt
    (e.g. 'sister', 'neighbor', 'colleague') to control for the model's
    prior preference among the three candidate names without any topical bias.
    """
    # Extract the relation word from the low-pressure prompt
    low = triple.pressure_low.lower()
    for rel in [
        "sister",
        "brother",
        "neighbor",
        "colleague",
        "friend",
        "aunt",
        "uncle",
        "cousin",
        "daughter",
        "son",
        "coworker",
        "cat",
        "dog",
        "grandmother",
        "grandfather",
        "dentist",
    ]:
        if rel in low:
            return BASELINE_TEMPLATE.format(relation=rel)
    # Fallback
    return BASELINE_TEMPLATE.format(relation="friend")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def load_triples(path: str) -> list[Triple]:
    """Load triples from JSONL."""
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                triples.append(Triple(**data))
    return triples


def run_experiment(
    triples: list[Triple],
    models: list[str],
    output_dir: pathlib.Path,
    delay: float = 13.0,
    dry_run: bool = False,
    include_baseline: bool = False,
) -> list[Result]:
    """Run all triples × pressure levels × models.

    If include_baseline is True, also runs a neutral-context probe per triple
    to measure the model's prior preference among the three candidate names
    without any semantic pressure.  This controls for unigram frequency
    confounds (hostile review flaw #3).
    """

    results: list[Result] = []
    pressure_levels = ["low", "mid", "high"]
    if include_baseline:
        pressure_levels = ["baseline"] + pressure_levels
    total = len(triples) * len(pressure_levels) * len(models)

    print(f"\n{'=' * 70}")
    print(f"  RigidBench v2 — Phonological-Semantic Asymmetry Experiment")
    print(
        f"  {len(triples)} triples × {len(pressure_levels)} pressure levels × {len(models)} models = {total} trials"
    )
    print(f"{'=' * 70}\n")

    trial = 0
    for model in models:
        print(f"\n{'─' * 70}")
        print(f"  MODEL: {model}")
        print(f"{'─' * 70}")

        model_results: list[Result] = []

        for triple in triples:
            for level in pressure_levels:
                trial += 1
                if level == "baseline":
                    prompt = make_baseline_prompt(triple)
                else:
                    prompt = getattr(triple, f"pressure_{level}")

                if dry_run:
                    print(
                        f"  [{trial}/{total}] {triple.id} @ {level:4s} | PROMPT: {prompt[:60]}..."
                    )
                    continue

                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        completion = query_model(model, prompt)
                        break
                    except Exception as e:
                        if "429" in str(e):
                            wait = 60
                            print(
                                f"  [{trial}/{total}] {triple.id} @ {level:4s} | Rate limit (429), waiting {wait}s (attempt {attempt + 1}/{max_retries})..."
                            )
                            time.sleep(wait)
                        else:
                            completion = f"[ERROR: {e}]"
                            print(
                                f"  [{trial}/{total}] {triple.id} @ {level:4s} | ERROR: {e}"
                            )
                            break
                else:
                    completion = "[ERROR: Max retries exceeded]"

                matched_word, error_type = classify_error(
                    completion,
                    triple.proper_noun,
                    triple.semantic_lure,
                    triple.phonological_neighbor,
                )

                result = Result(
                    triple_id=triple.id,
                    model=model,
                    pressure_level=level,
                    prompt=prompt,
                    proper_noun=triple.proper_noun,
                    semantic_lure=triple.semantic_lure,
                    phonological_neighbor=triple.phonological_neighbor,
                    raw_completion=completion,
                    matched_word=matched_word,
                    error_type=error_type,
                    phon_distance_name_to_lure=triple.phon_distance_name_to_lure,
                    phon_distance_name_to_neighbor=triple.phon_distance_name_to_neighbor,
                    semantic_sim_name_to_lure=triple.semantic_sim_name_to_lure,
                    semantic_sim_name_to_neighbor=triple.semantic_sim_name_to_neighbor,
                    etymological_link=triple.etymological_link,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                results.append(result)
                model_results.append(result)

                # Print status
                icon = {
                    "preserved": "✓",
                    "semantic_sub": "✗ SEM",
                    "phonological_sub": "✗ PHO",
                    "other": "? OTH",
                }[error_type]

                print(
                    f"  [{trial:3d}/{total}] {triple.id} @ {level:4s} | "
                    f"{triple.proper_noun:12s} → {matched_word:12s} {icon:8s} "
                    f"(raw: {completion[:40]})"
                )

                time.sleep(delay)

        # Per-model summary
        if model_results:
            _print_model_summary(model, model_results)

    # Save results
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "rigidbench_v2_results.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        print(f"\n  Results saved to {results_path}")

    return results


def _print_model_summary(model: str, results: list[Result]) -> None:
    """Print per-model aggregate stats."""
    total = len(results)
    preserved = sum(1 for r in results if r.error_type == "preserved")
    semantic = sum(1 for r in results if r.error_type == "semantic_sub")
    phonological = sum(1 for r in results if r.error_type == "phonological_sub")
    other = sum(1 for r in results if r.error_type == "other")

    print(f"\n  ┌────────────────────────── {model} ──────────────────────────┐")
    print(
        f"  │ Preserved identity:    {preserved:3d}/{total} ({preserved / total * 100:5.1f}%)                    │"
    )
    print(
        f"  │ Semantic substitution: {semantic:3d}/{total} ({semantic / total * 100:5.1f}%)  ← THE KEY METRIC     │"
    )
    print(
        f"  │ Phonological sub:      {phonological:3d}/{total} ({phonological / total * 100:5.1f}%)                    │"
    )
    print(
        f"  │ Other:                 {other:3d}/{total} ({other / total * 100:5.1f}%)                    │"
    )
    print(f"  └────────────────────────────────────────────────────────────┘")

    # Per-pressure-level breakdown
    for level in ["low", "mid", "high"]:
        level_results = [r for r in results if r.pressure_level == level]
        if not level_results:
            continue
        lt = len(level_results)
        lp = sum(1 for r in level_results if r.error_type == "preserved")
        ls = sum(1 for r in level_results if r.error_type == "semantic_sub")
        lph = sum(1 for r in level_results if r.error_type == "phonological_sub")
        lo = sum(1 for r in level_results if r.error_type == "other")
        print(
            f"    {level:4s}: preserved={lp}/{lt} "
            f"semantic={ls}/{lt} "
            f"phonological={lph}/{lt} "
            f"other={lo}/{lt}"
        )


# ---------------------------------------------------------------------------
# v3.1 item loading
# ---------------------------------------------------------------------------

# Canonical family names used in v3.1 JSONL files
V3_FAMILIES = {
    "completion_under_pressure",  # Family A
    "multi_turn_persistence",  # Family B
    "summary_compression",  # Family C
    "clarify_abstain",  # Family D
    "entity_set_competition",  # Family E
}

# Short aliases for CLI --families flag
_FAMILY_ALIASES: dict[str, str] = {
    "a": "completion_under_pressure",
    "b": "multi_turn_persistence",
    "c": "summary_compression",
    "d": "clarify_abstain",
    "e": "entity_set_competition",
}


def load_v3_items(
    paths: list[str],
    families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load v3.1 items from one or more JSONL files.

    Each line must be a JSON object with at least ``id``, ``family``,
    ``mention``, ``semantic_lure``, and ``prompt_turns`` fields.

    If *families* is provided, only items whose ``family`` value is in
    the set are returned.
    """
    items: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                fam = item.get("family", "")
                if families and fam not in families:
                    continue
                items.append(item)
    return items


# ---------------------------------------------------------------------------
# v3.1 experiment loop
# ---------------------------------------------------------------------------


def _v3_score_for_family(outcome: str, family: str) -> float:
    """Return numeric score (0.0 / 0.5 / 1.0) per v3.1 scoring rules."""
    if family in (
        "completion_under_pressure",
        "multi_turn_persistence",
        "summary_compression",
    ):
        if outcome in ("PRES", "ALIAS_OK"):
            return 1.0
        if outcome == "CLARIFY":
            return 0.5
        return 0.0
    elif family == "clarify_abstain":
        if outcome in ("CLARIFY", "ABSTAIN"):
            return 1.0
        if outcome == "PRES":
            return 0.5
        return 0.0
    elif family == "entity_set_competition":
        if outcome in ("PRES", "ALIAS_OK"):
            return 1.0
        if outcome == "CLARIFY":
            return 0.5
        return 0.0
    return 0.0


def run_v3_experiment(
    items: list[dict[str, Any]],
    models: list[str],
    output_dir: pathlib.Path,
    delay: float = 13.0,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run v3.1 items across models.

    Each item is dispatched to the appropriate handler based on its
    ``family`` field.  Results are written as JSONL with a ``family``
    field for downstream analysis.
    """
    results: list[dict[str, Any]] = []
    total = len(items) * len(models)

    # Group items by family for the banner
    family_counts: dict[str, int] = {}
    for item in items:
        fam = item.get("family", "unknown")
        family_counts[fam] = family_counts.get(fam, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"  RigidBench v3.1 — Relational Invariance Experiment")
    print(f"  {len(items)} items × {len(models)} models = {total} trials")
    for fam, count in sorted(family_counts.items()):
        print(f"    {fam}: {count} items")
    print(f"{'=' * 70}\n")

    results_path = output_dir / "rigidbench_v3_results.jsonl"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            results_path.unlink()

    trial = 0
    for model in models:
        print(f"\n{'─' * 70}")
        print(f"  MODEL: {model}")
        print(f"{'─' * 70}")

        model_results: list[dict[str, Any]] = []

        for item in items:
            trial += 1
            item_id = item.get("id", "???")
            family = item.get("family", "unknown")
            mention = item.get("mention", "")
            pressure = item.get("pressure_level", "high")

            # Build messages
            messages = build_v3_messages(item)
            max_tokens = get_v3_max_tokens(family)

            # For dry-run, show first message preview
            if dry_run:
                first_msg = messages[0]["content"][:60]
                print(
                    f"  [{trial}/{total}] {item_id} ({family}) | PROMPT: {first_msg}..."
                )
                continue

            # Query with retries
            max_retries = 5
            completion = ""
            for attempt in range(max_retries):
                try:
                    completion = query_model_multiturn(
                        model, messages, max_tokens=max_tokens
                    )
                    break
                except Exception as e:
                    if "429" in str(e):
                        wait = 60
                        print(
                            f"  [{trial}/{total}] {item_id} | "
                            f"Rate limit (429), waiting {wait}s "
                            f"(attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait)
                    else:
                        completion = f"[ERROR: {e}]"
                        print(f"  [{trial}/{total}] {item_id} | ERROR: {e}")
                        break
            else:
                completion = "[ERROR: Max retries exceeded]"

            # Classify outcome
            matched_word, outcome = classify_v3_outcome(completion, item)
            score = _v3_score_for_family(outcome, family)

            # Build result dict (superset of Result fields + v3.1 extras)
            result = {
                "triple_id": item_id,
                "model": model,
                "family": family,
                "pressure_level": pressure,
                "pressure_operator": item.get("pressure_operator", ""),
                "primary_relation": item.get("primary_relation", ""),
                "prompt": messages[-1]["content"] if messages else "",
                "prompt_turn_count": len(item.get("prompt_turns", [])),
                "proper_noun": mention,
                "semantic_lure": item.get("semantic_lure", ""),
                "phonological_neighbor": item.get("phonological_neighbor", ""),
                "raw_completion": completion,
                "matched_word": matched_word,
                "error_type": outcome,
                "score": score,
                "phon_distance_name_to_lure": item.get(
                    "phonological_distance_name_lure", 0
                ),
                "phon_distance_name_to_neighbor": item.get(
                    "phonological_distance_name_neighbor", 0
                ),
                "semantic_sim_name_to_lure": item.get(
                    "semantic_similarity_name_lure", 0.0
                ),
                "semantic_sim_name_to_neighbor": 0.0,
                "etymological_link": item.get("etymological_link", False),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            results.append(result)
            model_results.append(result)
            if not dry_run:
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

            # Status icon
            icon_map = {
                "PRES": "✓",
                "ALIAS_OK": "✓ ALI",
                "SEM_SUB": "✗ SEM",
                "PHO_SUB": "✗ PHO",
                "CLARIFY": "? CLR",
                "ABSTAIN": "— ABS",
                "ENT_CONF": "✗ ENT",
                "META": "⊘ MET",
                "NOISE": "? NOI",
            }
            icon = icon_map.get(outcome, "?")

            print(
                f"  [{trial:3d}/{total}] {item_id} ({family[:8]:8s}) "
                f"@ {pressure:4s} | "
                f"{mention:12s} → {matched_word:12s} {icon:8s} "
                f"(raw: {completion[:40]})"
            )

            time.sleep(delay)

        # Per-model summary
        if model_results:
            _print_v3_model_summary(model, model_results)

    # Save results
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n  Results saved to {results_path}")

    return results


def _print_v3_model_summary(model: str, results: list[dict[str, Any]]) -> None:
    """Print per-model aggregate stats for v3.1 results."""
    total = len(results)
    by_outcome: dict[str, int] = {}
    for r in results:
        ot = r["error_type"]
        by_outcome[ot] = by_outcome.get(ot, 0) + 1

    pres = by_outcome.get("PRES", 0) + by_outcome.get("ALIAS_OK", 0)
    sem = by_outcome.get("SEM_SUB", 0)
    pho = by_outcome.get("PHO_SUB", 0)
    clarify = by_outcome.get("CLARIFY", 0)
    abstain = by_outcome.get("ABSTAIN", 0)
    ent_conf = by_outcome.get("ENT_CONF", 0)
    other = total - pres - sem - pho - clarify - abstain - ent_conf

    # IPR and RDR
    ipr = pres / total * 100 if total else 0
    sub_total = sem + pho
    rdr = sem / sub_total if sub_total > 0 else float("nan")

    avg_score = sum(r["score"] for r in results) / total if total else 0

    print(f"\n  ┌─────────────── {model} (v3.1) ───────────────┐")
    print(f"  │ IPR (identity preservation): {pres:3d}/{total} ({ipr:5.1f}%)     │")
    print(f"  │ SEM_SUB:  {sem:3d}   PHO_SUB: {pho:3d}   RDR: {rdr:5.2f}      │")
    print(
        f"  │ CLARIFY:  {clarify:3d}   ABSTAIN: {abstain:3d}   ENT_CONF: {ent_conf:3d} │"
    )
    print(f"  │ Other:    {other:3d}   Avg score: {avg_score:.3f}            │")
    print(f"  └──────────────────────────────────────────────┘")

    # Per-family breakdown
    families_seen: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        fam = r["family"]
        families_seen.setdefault(fam, []).append(r)

    for fam, fam_results in sorted(families_seen.items()):
        ft = len(fam_results)
        fp = sum(1 for r in fam_results if r["error_type"] in ("PRES", "ALIAS_OK"))
        fs = sum(1 for r in fam_results if r["error_type"] == "SEM_SUB")
        fc = sum(1 for r in fam_results if r["error_type"] in ("CLARIFY", "ABSTAIN"))
        favg = sum(r["score"] for r in fam_results) / ft if ft else 0
        print(
            f"    {fam:30s}: "
            f"PRES={fp}/{ft} SEM={fs}/{ft} "
            f"CLR/ABS={fc}/{ft} avg={favg:.2f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_smoke_test(models: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print("  Backend Smoke Test")
    print(f"{'=' * 70}\n")
    for model in models:
        print(f"Testing model: {model}")
        prompt = "Reply with exactly OK."
        try:
            response = query_model(model, prompt)
            print(f"  Response: {response}")
            if response.strip() in ("OK", "OK."):
                print("  Status: SUCCESS")
            else:
                print("  Status: WARNING (Did not reply with exactly OK)")
        except Exception as e:
            print(f"  Status: ERROR ({e})")
        print()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="RigidBench v2/v3.1 experiment runner")
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(
            pathlib.Path(__file__).parent.parent
            / "benchmark"
            / "rigidbench_v2_triples.jsonl"
        ),
        help="Path to v2 triples JSONL (Family A legacy mode)",
    )
    parser.add_argument(
        "--v3-items",
        type=str,
        nargs="*",
        default=None,
        help="Paths to v3.1 JSONL files.  When provided, runs the v3.1 "
        "experiment loop instead of the legacy v2 loop.",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="*",
        default=None,
        help="Filter v3.1 items to these families.  Accepts full names "
        "(e.g. multi_turn_persistence) or short aliases (a/b/c/d/e).  "
        "Default: all families.",
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model names")
    parser.add_argument(
        "--thinking", type=str, help="Thinking config (off, budget_tokens=X)"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=13.0,
        help="Seconds between API calls (13s = safe for 5 RPM free tier)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview prompts without running"
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="(v2 only) Also run a neutral-context probe per triple to "
        "measure prior name-frequency preference",
    )
    parser.add_argument(
        "--backend-smoke-test",
        action="store_true",
        help="Run a smoke test on the specified models and exit",
    )
    args = parser.parse_args()

    if args.backend_smoke_test:
        run_smoke_test(args.models)

    # --- v3.1 mode ---
    if args.v3_items is not None:
        # Resolve family filter
        family_filter: set[str] | None = None
        if args.families:
            family_filter = set()
            for f in args.families:
                resolved = _FAMILY_ALIASES.get(f.lower(), f)
                family_filter.add(resolved)

        # If --v3-items given with no paths, auto-discover benchmark dir
        v3_paths = args.v3_items
        if not v3_paths:
            bench_dir = pathlib.Path(__file__).parent.parent / "benchmark"
            v3_paths = sorted(str(p) for p in bench_dir.glob("rigidbench_v3_*.jsonl"))
            if not v3_paths:
                print("ERROR: No v3.1 JSONL files found in", bench_dir)
                sys.exit(1)

        items = load_v3_items(v3_paths, families=family_filter)
        if not items:
            print("ERROR: No items loaded (check --families filter)")
            sys.exit(1)
        print(f"Loaded {len(items)} v3.1 items from {len(v3_paths)} file(s)")
        if family_filter:
            print(f"  Family filter: {sorted(family_filter)}")

        run_v3_experiment(
            items=items,
            models=args.models,
            output_dir=pathlib.Path(args.output),
            delay=args.delay,
            dry_run=args.dry_run,
        )
        return

    # --- Legacy v2 mode ---
    triples = load_triples(args.prompts)
    print(f"Loaded {len(triples)} triples from {args.prompts}")

    run_experiment(
        triples=triples,
        models=args.models,
        output_dir=pathlib.Path(args.output),
        delay=args.delay,
        dry_run=args.dry_run,
        include_baseline=args.include_baseline,
    )


if __name__ == "__main__":
    main()
