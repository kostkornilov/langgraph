"""Centralized embedding provider helpers for the LangGraph RAG stack.

This module ensures the same embedding provider (Google GenAI by default)
is used consistently across ingestion/build scripts and at query time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Literal, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import OpenAIEmbeddings

ProviderLiteral = Literal["google-genai", "sentence-transformers", "openai"]


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: ProviderLiteral
    model: str
    normalize_l2: bool = True
    similarity_metric: Literal["ip", "l2"] = "ip"


class EmbeddingProvider:
    """Adapter that exposes embed_documents/embed_query plus config metadata."""

    def __init__(
        self,
        provider: ProviderLiteral,
        model: str,
        embed_documents_fn: Callable[[Iterable[str]], Iterable[Iterable[float]]],
        embed_query_fn: Callable[[str], Iterable[float]],
        normalize_l2: bool = True,
        similarity_metric: Literal["ip", "l2"] = "ip",
    ) -> None:
        self.provider = provider
        self.model = model
        self._embed_documents = embed_documents_fn
        self._embed_query = embed_query_fn
        self.normalize_l2 = normalize_l2
        self.similarity_metric = similarity_metric

    def embed_documents(self, texts: Iterable[str]):
        return self._embed_documents(list(texts))

    def embed_query(self, text: str):
        return self._embed_query(text)

    def as_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.provider,
            model=self.model,
            normalize_l2=self.normalize_l2,
            similarity_metric=self.similarity_metric,
        )


from typing import Tuple


def _build_google_provider(task: str) -> Tuple[str, "GoogleGenerativeAIEmbeddings"]:
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; cannot use Google embeddings")

    model_name = os.environ.get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    return model_name, GoogleGenerativeAIEmbeddings(
        model=model_name,
        task_type=task,
        google_api_key=api_key,
    )


def _sentence_transformer_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("SENTENCE_TRANSFORMERS_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    return SentenceTransformer(model_name)


def _openai_embeddings() -> "OpenAIEmbeddings":
    from langchain.embeddings import OpenAIEmbeddings

    return OpenAIEmbeddings()


@lru_cache(maxsize=1)
def _cached_sentence_transformer():
    return _sentence_transformer_model()


def get_embedding_provider(
    *,
    require_google: bool = False,
    expected_provider: Optional[ProviderLiteral] = None,
    expected_model: Optional[str] = None,
) -> EmbeddingProvider:
    """Return an EmbeddingProvider, preferring Google GenAI embeddings.

    Parameters
    ----------
    require_google: bool
        If True, raise immediately when GOOGLE_API_KEY is missing.
    expected_provider: Optional[str]
        If provided, ensure the resolved provider matches (used when loading stores).
    expected_model: Optional[str]
        Soft check that the chosen model matches stored config; logs mismatch but continues.
    """

    errors: Dict[str, str] = {}

    # 1. Google GenAI embeddings (preferred for multilingual Russian RAG)
    allow_fallback = os.environ.get("ALLOW_EMBEDDING_FALLBACK") in {"1", "true", "True"}

    try:
        model_name, google_doc = _build_google_provider("retrieval_document")
        _, google_query = _build_google_provider("retrieval_query")
        provider = EmbeddingProvider(
            provider="google-genai",
            model=model_name,
            embed_documents_fn=google_doc.embed_documents,
            embed_query_fn=google_query.embed_query,
            normalize_l2=True,
            similarity_metric="ip",
        )
        if expected_provider and expected_provider != provider.provider:
            raise RuntimeError(
                f"Expected embedding provider {expected_provider} but google-genai is configured"
            )
        if expected_model and expected_model != provider.model:
            print(
                f"[embedding] Warning: stored model {expected_model} differs from configured {provider.model}."
            )
        return provider
    except Exception as exc:
        errors["google-genai"] = str(exc)
        if require_google and not allow_fallback:
            raise RuntimeError(
                "Google embeddings required but unavailable. Set GOOGLE_API_KEY and install langchain-google-genai."
            ) from exc

    # 2. SentenceTransformers fallback (CPU-friendly multilingual)
    try:
        model = _cached_sentence_transformer()
        model_name = getattr(model, "model_card_data", {}).get("model_name") or getattr(model, "_model_card_name", None) or os.environ.get("SENTENCE_TRANSFORMERS_MODEL", "sentence-transformer")
        provider = EmbeddingProvider(
            provider="sentence-transformers",
            model=model_name,
            embed_documents_fn=lambda texts: model.encode(list(texts), show_progress_bar=True),
            embed_query_fn=lambda text: model.encode([text], show_progress_bar=False)[0],
            normalize_l2=True,
            similarity_metric="ip",
        )
        if expected_provider and expected_provider != provider.provider:
            raise RuntimeError(
                f"Expected embedding provider {expected_provider} but sentence-transformers is configured"
            )
        if expected_model and expected_model not in provider.model:
            print(
                f"[embedding] Warning: stored model {expected_model} differs from configured {provider.model}."
            )
        return provider
    except Exception as exc:
        errors["sentence-transformers"] = str(exc)

    # 3. OpenAI fallback (only if key available)
    try:
        openai_emb = _openai_embeddings()
        provider = EmbeddingProvider(
            provider="openai",
            model="text-embedding-3-small",
            embed_documents_fn=openai_emb.embed_documents,
            embed_query_fn=openai_emb.embed_query,
            normalize_l2=True,
            similarity_metric="ip",
        )
        if expected_provider and expected_provider != provider.provider:
            raise RuntimeError(
                f"Expected embedding provider {expected_provider} but openai is configured"
            )
        return provider
    except Exception as exc:
        errors["openai"] = str(exc)

    raise RuntimeError(
        "No embedding provider available. Errors: " + ", ".join(f"{k}: {v}" for k, v in errors.items())
    )


def save_embedding_config(path: os.PathLike, config: EmbeddingConfig) -> None:
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config.__dict__, fh, ensure_ascii=False, indent=2)


def load_embedding_config(path: os.PathLike) -> EmbeddingConfig:
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return EmbeddingConfig(**data)
