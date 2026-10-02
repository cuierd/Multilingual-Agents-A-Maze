from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .components import attach_punctuation, strip_punctuation


def sorted_distractor_keys(out: Mapping[str, Any]) -> List[str]:
    keyed: List[tuple[int, str]] = []
    for key in out.keys():
        if not str(key).startswith("distractor"):
            continue
        suffix = str(key)[len("distractor") :]
        if suffix.isdigit():
            keyed.append((int(suffix), str(key)))
    keyed.sort(key=lambda x: x[0])
    return [k for _, k in keyed]


def extract_forced_target(core: str) -> Optional[str]:
    """
    Detect marker tokens of form **target** (optionally **{target}**).
    Returns inner target string when marker is present, else None.
    """
    text = str(core or "").strip()
    if len(text) < 4 or not (text.startswith("**") and text.endswith("**")):
        return None
    inner = text[2:-2].strip()
    if inner.startswith("{") and inner.endswith("}") and len(inner) >= 2:
        inner = inner[1:-1].strip()
    return inner


def reattach_punctuation_to_output(
    out: Dict[str, Any],
    *,
    prefix: str,
    suffix: str,
) -> Dict[str, Any]:
    patched = dict(out)

    def wrap(x: Any) -> Any:
        if x is None:
            return None
        return attach_punctuation(str(x), prefix, suffix)

    for key in ("source", *sorted_distractor_keys(patched)):
        if key in patched:
            patched[key] = wrap(patched[key])
    return patched


def match_capitalization(text: str, reference: str) -> str:
    """
    Match the capitalization pattern of `reference` for simple word-like strings.

    Examples:
    reference = "Haus"  -> "katze" becomes "Katze"
    reference = "NASA"  -> "test" becomes "TEST"
    reference = "haus"  -> "Katze" becomes "katze"
    """
    text = str(text or "")
    reference = str(reference or "")

    if not text:
        return text

    if reference.isupper():
        return text.upper()

    if reference.islower():
        return text.lower()

    if reference[:1].isupper():
        return text[:1].upper() + text[1:]

    return text


def ensure_distractor_count(
    out: Dict[str, Any],
    *,
    num_distractors: int,
    target_word: str,
    target_core: str,
    candidate_pool: Sequence[str],
    puncts: Set[str],
    allow_candidate_fill: bool = True,
) -> Dict[str, Any]:
    """Normalize output to exactly num_distractors fields."""
    n = int(num_distractors)
    if n <= 0:
        raise ValueError("num_distractors must be positive")

    dummy_len = max(1, len(str(target_core or target_word or "")))
    dummy = match_capitalization("X" * dummy_len, target_core)

    base = {k: v for k, v in out.items() if not str(k).startswith("distractor")}

    selected: List[str] = []
    seen_cores: Set[str] = set()
    target_norm = str(target_core or "").strip().casefold()

    def push(value: Any) -> None:
        if value is None:
            return

        text = str(value).strip()
        if not text:
            return

        pre, core, suf = strip_punctuation(text, puncts)
        core = str(core or "").strip()

        if not core:
            return

        core_norm = core.casefold()

        if target_norm and core_norm == target_norm:
            return

        if core_norm in seen_cores:
            return

        seen_cores.add(core_norm)

        matched_core = match_capitalization(core, target_core)
        selected.append(attach_punctuation(matched_core, pre, suf))

    for key in sorted_distractor_keys(out):
        push(out.get(key))

    if allow_candidate_fill:
        for cand in candidate_pool:
            push(cand)
            if len(selected) >= n:
                break

    while len(selected) < n:
        selected.append(dummy)

    for i in range(n):
        base[f"distractor{i + 1}"] = selected[i]

    return base


def build_dummy_output(
    *,
    source: str,
    dummy_len: int,
    num_distractors: int,
) -> Dict[str, Any]:
    dummy = "X" * max(1, int(dummy_len))
    out: Dict[str, Any] = {"source": str(source)}
    for i in range(int(num_distractors)):
        out[f"distractor{i + 1}"] = dummy
    return out

