from collections.abc import Generator

import ollama as ollama_sdk
import structlog

from backend.config import settings

logger = structlog.get_logger()


def get_client() -> ollama_sdk.Client:
    return ollama_sdk.Client(host=settings.ollama_host)


def list_models() -> list[str]:
    try:
        client = get_client()
        models = client.list()
        return [m.model for m in models.models]
    except Exception as e:
        logger.warning("llm.list_models_failed", error=str(e))
        return []


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Synchronous chat completion via Ollama."""
    client = get_client()
    model_name = model or settings.ollama_model

    response = client.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    return response.message.content


def chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """Streaming chat completion via Ollama. Yields token chunks."""
    client = get_client()
    model_name = model or settings.ollama_model

    stream = client.chat(
        model=model_name,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    for chunk in stream:
        token = chunk.message.content
        if token:
            yield token


def generate(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Simple generate with optional system prompt."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
