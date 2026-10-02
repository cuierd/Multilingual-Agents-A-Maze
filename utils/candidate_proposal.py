"""
Shared distractor candidate construction for Chat and HF (LLMAgent) pipelines.

Supports:
  - lexicon neighborhoods (frequency-binned neighbors)
  - model-proposed pools (parsed from JSON / json_repair lenient schemas)
"""

from __future__ import annotations

import json

from json_repair import repair_json
from collections.abc import Callable, Sequence
from typing import Any

from .components import Lexicon, normalize_candidates


def extract_hf_generated_text(resp: Any) -> str:
    """Normalize transformers / LangChain HuggingFace pipeline outputs to plain text."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("generated_text", "text", "content"):
            if key in resp and resp[key] is not None:
                return str(resp[key])
    if isinstance(resp, list):
        if not resp:
            return ""
        first = resp[0]
        if isinstance(first, dict):
            return extract_hf_generated_text(first)
        if isinstance(first, list):
            return extract_hf_generated_text(first[0] if first else "")
    return str(resp)


def parse_distractor_pool_lenient(text: str) -> list[str]:
    """Parse ``{\"distractors\": [...]}`` or a raw list; return strings only ([] on failure)."""
    try:
        fixed = repair_json(text)
        obj = json.loads(fixed)
        words = obj.get("distractors") if isinstance(obj, dict) else obj if isinstance(obj, list) else []
        return [w for w in words if isinstance(w, str)]
    except Exception:
        return []


def finalize_candidate_words(
    words: Sequence[str],
    core: str,
    *,
    min_candidates: int,
    max_candidates: int,
) -> list[str]:
    """
    Drop empty / duplicate / target-equal entries, clip, then pad to ``min_candidates`` if needed.
    Padding uses repeated ``\"X\" * len(core)`` tokens.
    """
    cleaned = normalize_candidates(list(words))
    cleaned = [c for c in cleaned if c and c.strip().lower() != core.strip().lower()]
    cleaned = cleaned[: int(max_candidates)]
    pad_token = "X" * max(1, len(core))
    if len(cleaned) < int(min_candidates):
        pad_count = max(0, int(min_candidates) - len(cleaned))
        cleaned = (cleaned + [pad_token] * pad_count)[: int(max_candidates)]
    return cleaned


class LexiconCandidateCache:
    """Cache lexicon neighborhoods per stripped word core."""
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon
        self._neighbor_cache: dict[tuple[str, int, int], tuple[str, ...]] = {}

    def neighbors(self, core: str, *, min_candidates: int, max_candidates: int) -> tuple[str, ...]:
        key = (core, int(min_candidates), int(max_candidates))

        if key not in self._neighbor_cache:
            raw = self.lexicon.get_neighbor(
                core,
                min_size=int(min_candidates),
                max_size=int(max_candidates),
            )
            clean = normalize_candidates(raw)
            clean = [w for w in clean if w and w.strip().lower() != core.strip().lower()]
            self._neighbor_cache[key] = tuple(clean)

        return self._neighbor_cache[key]


def lexicon_candidates(
    cache: LexiconCandidateCache,
    core: str,
    *,
    min_candidates: int,
    max_candidates: int,
) -> list[str]:
    cands = list(cache.neighbors(core, min_candidates=min_candidates, max_candidates=max_candidates))
    return finalize_candidate_words(cands, core, min_candidates=min_candidates, max_candidates=max_candidates)


def invoke_openai_generator(
    *,
    render_messages: Callable[..., Any],
    chat_model: Any,
    sentence_prefix: str,
    core: str,
) -> list[str]:
    """Run Stage-A OpenAI messages and return raw word strings (before finalize)."""
    gen_msgs = render_messages(sentence_prefix=sentence_prefix, word=core)
    gen_resp = chat_model.invoke(gen_msgs)
    text = getattr(gen_resp, "content", str(gen_resp))
    return parse_distractor_pool_lenient(text)


def apply_model_proposed_candidates(
    raw_words: Sequence[str],
    core: str,
    *,
    min_candidates: int,
    max_candidates: int,
) -> list[str]:
    return finalize_candidate_words(raw_words, core, min_candidates=min_candidates, max_candidates=max_candidates)


def resolve_surprisal_model_id(*, primary_model_id: str, selection_model_id: str | None = None) -> str:
    """Prefer explicit selection model for surprisal scoring when provided."""
    sel = (selection_model_id or "").strip()
    return sel or str(primary_model_id)
