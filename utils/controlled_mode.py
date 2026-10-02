from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

from .components import Lexicon, attach_punctuation, join_tokens, normalize_candidates, strip_punctuation
from .distractor_utils import build_dummy_output, extract_forced_target, match_capitalization


def split_controlled_sentence(sentence: str, split_on: Optional[str]) -> List[str]:
    text = str(sentence or "").strip()
    if not text:
        return []
    if split_on is not None and str(split_on) != "":
        return [t for t in text.split(split_on) if t]
    return text.split()


def read_controlled_input(path: Union[str, Path], split_on: Optional[str]) -> List[Dict[str, Any]]:
    """
    Read controlled-experiment rows. Header aliases are supported, e.g.:
    - item: item_id/item/itemid
    - condition: condition_id/condition/cond/cond_id
    - sentence: sentence/stimuli/stimulus/text/sent
    """
    path = Path(path)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    item_map: Dict[str, List[Dict[str, Any]]] = {}
    column_idx: Optional[Dict[str, int]] = None

    item_aliases = {"item_id", "item", "itemid"}
    cond_aliases = {"condition_id", "conditionid", "condition", "cond", "cond_id"}
    sent_aliases = {"sentence", "stimuli", "stimulus", "text", "sent"}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not str(x).strip() for x in row):
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Controlled input row {row_idx} must have at least 3 columns: "
                    f"item_id, condition_id, sentence."
                )

            if row_idx == 1:
                header = [str(c).strip().lower() for c in row]

                def _index_of(aliases: set[str]) -> Optional[int]:
                    for i, name in enumerate(header):
                        if name in aliases:
                            return i
                    return None

                i_item = _index_of(item_aliases)
                i_cond = _index_of(cond_aliases)
                i_sent = _index_of(sent_aliases)
                if i_item is not None and i_cond is not None and i_sent is not None:
                    column_idx = {"item": i_item, "cond": i_cond, "sent": i_sent}
                    continue

            if column_idx is not None:
                item_id = str(row[column_idx["item"]]).strip()
                condition_id = str(row[column_idx["cond"]]).strip()
                sentence = str(row[column_idx["sent"]]).strip()
            else:
                item_id = str(row[0]).strip()
                condition_id = str(row[1]).strip()
                sentence = delimiter.join(row[2:]).strip()

            if not item_id or not condition_id:
                raise ValueError(f"Controlled input row {row_idx} has empty item_id or condition_id.")
            tokens = split_controlled_sentence(sentence, split_on)
            if not tokens:
                raise ValueError(f"Controlled input row {row_idx} has empty tokenized sentence.")
            item_map.setdefault(item_id, []).append(
                {"condition_id": condition_id, "sentence": sentence, "tokens": tokens}
            )

    if not item_map:
        raise ValueError("Controlled input is empty after parsing.")

    items: List[Dict[str, Any]] = []
    for item_id, rows in item_map.items():
        lengths = [len(r["tokens"]) for r in rows]
        if len(set(lengths)) != 1:
            details = ", ".join(f"{r['condition_id']}:{len(r['tokens'])}" for r in rows)
            raise ValueError(
                f"Item '{item_id}' has mismatched sentence lengths across conditions: {details}."
            )
        items.append({"item_id": item_id, "conditions": rows})
    return items


class ControlledModeService:
    """Within-item controlled Maze: shared distractor output per (item, word index).
    Candidate pools come from lexicon neighborhoods via ``get_candidates`` (controlled experiments
    require lexicon-backed candidates)!
    """

    def __init__(
        self,
        *,
        lexicon: Optional[Lexicon],
        puncts: Set[str],
        num_distractors: int,
        token_joiner: str,
        min_candidates: int,
        max_candidates: int,
        apply_surprisal_threshold: bool,
        threshold_filter: Any,
        get_candidates: Callable[[str], tuple[str, str, str, Sequence[str], Optional[int]]],
        invoke_raw: Callable[[str, str, Sequence[str]], str],
        parse_to_dict: Callable[[str, bool], Dict[str, Any]],
        ensure_distractor_count: Callable[..., Dict[str, Any]],
        reattach_punct: Callable[[Dict[str, Any], str, str], Dict[str, Any]],
        sorted_distractor_keys: Callable[[Dict[str, Any]], List[str]],
    ) -> None:
        self.lexicon = lexicon
        self.puncts = puncts
        self.num_distractors = int(num_distractors)
        self.token_joiner = token_joiner
        self.min_candidates = int(min_candidates)
        self.max_candidates = int(max_candidates)
        self.apply_surprisal_threshold = bool(apply_surprisal_threshold)
        self.threshold_filter = threshold_filter
        self.get_candidates = get_candidates
        self.invoke_raw = invoke_raw
        self.parse_to_dict = parse_to_dict
        self.ensure_distractor_count = ensure_distractor_count
        self.reattach_punct = reattach_punct
        self.sorted_distractor_keys = sorted_distractor_keys

    def _token_metadata(self, token: str) -> tuple[str, str, str, Optional[int]]:
        """Strip punctuation and forced-target markers without building lexicon/model pools."""
        pfx, core, sfx = strip_punctuation(str(token), self.puncts)
        if not core:
            return pfx, core, sfx, None
        forced = extract_forced_target(core)
        if forced is not None:
            ft = str(forced).strip()
            return pfx, ft, sfx, max(1, len(ft))
        return pfx, core, sfx, None

    @staticmethod
    def _slot_needs_per_condition_candidate_pools(
        word_index: int,
        token_metas: Sequence[tuple[str, str, str, Optional[int]]],
    ) -> bool:
        """
        True only when :meth:`_process_slot` uses per-variant ``candidates`` (single shared core,
        non-initial slot, no forced dummy). Dummy and profile branches ignore neighbor pools.
        """
        if word_index == 0:
            return False
        if any(m[3] is not None for m in token_metas):
            return False
        cores = {str(c).strip() for _, c, _, _ in token_metas if str(c).strip()}
        return len(cores) <= 1

    @staticmethod
    def _compose_shared_source(cores: Sequence[str]) -> str:
        uniq: List[str] = []
        seen = set()
        for c in cores:
            c = str(c or "").strip()
            if not c or c in seen:
                continue
            seen.add(c)
            uniq.append(c)
        if not uniq:
            return ""
        if len(uniq) == 1:
            return uniq[0]
        return "|".join(uniq)

    @staticmethod
    def _shared_punct(variants: Sequence[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
        prefixes = {str(v.get("punct_prefix", "")) for v in variants}
        suffixes = {str(v.get("punct_suffix", "")) for v in variants}
        if len(prefixes) == 1 and len(suffixes) == 1:
            return next(iter(prefixes)), next(iter(suffixes))
        return None, None

    def _select_profile_candidates(self, variants: Sequence[Dict[str, Any]]) -> tuple[List[str], int, int]:
        if self.lexicon is None:
            raise RuntimeError(
                "Controlled mode requires a lexicon for profile-based neighbor pooling when targets differ across conditions."
            )
        cores = [str(v["target_core"]) for v in variants if str(v.get("target_core", "")).strip()]
        if not cores:
            return [], 1, int(self.lexicon.max_frequency_rank)
        avg_len = max(1, int(round(mean([len(c) for c in cores]))))
        ranks = [
            int(self.lexicon.get_rank(c, default_to_max=True) or self.lexicon.max_frequency_rank)
            for c in cores
        ]
        avg_rank = max(1, int(round(mean(ranks))))
        candidates = self.lexicon.get_neighbor_by_profile(
            target_length=avg_len,
            target_rank=avg_rank,
            min_size=self.min_candidates,
            max_size=self.max_candidates,
            exclude_words=set(cores),
        )
        candidates = normalize_candidates(candidates)
        core_set = set(cores)
        candidates = [c for c in candidates if c not in core_set]
        return candidates, avg_len, avg_rank

    def _enforce_avg_surprisal_threshold(
        self,
        out: Dict[str, Any],
        *,
        variants: Sequence[Dict[str, Any]],
        candidate_pool: Sequence[str],
    ) -> Dict[str, Any]:
        scorer = getattr(self.threshold_filter, "scorer", None)
        if not self.apply_surprisal_threshold or scorer is None:
            return out

        surprisal_cache: Dict[tuple[str, str], Optional[float]] = {}

        def surprisal_cached(context_prefix: str, word: str) -> Optional[float]:
            key = (context_prefix, word)
            if key in surprisal_cache:
                return surprisal_cache[key]
            try:
                val = float(
                    scorer.word_surprisal(context_prefix, word, token_joiner=self.token_joiner)
                )
            except Exception:
                val = None
            surprisal_cache[key] = val
            return val

        target_scores: List[float] = []
        target_cores = {str(v.get("target_core", "")).strip() for v in variants}
        target_cores_cf = {c.casefold() for c in target_cores if c}
        ref_core = next(
            (c for c in (str(v.get("target_core", "")).strip() for v in variants) if c),
            "",
        )
        for v in variants:
            core = str(v.get("target_core", "")).strip()
            if not core:
                continue
            s = surprisal_cached(str(v.get("context_prefix", "")), core)
            if s is not None:
                target_scores.append(float(s))
        if not target_scores:
            return out

        target_avg = float(mean(target_scores))
        if not math.isfinite(target_avg):
            return out

        required = float(self.threshold_filter._required_surprisal(target_avg))
        if not math.isfinite(required):
            return out

        cand_scores: Dict[str, float] = {}
        for cand in candidate_pool:
            vals: List[float] = []
            cw = str(cand)
            for v in variants:
                s = surprisal_cached(str(v.get("context_prefix", "")), cw)
                if s is not None:
                    vals.append(float(s))
            if vals:
                cand_scores[cw] = float(mean(vals))

        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        fallback_pool = [
            w
            for w, s in ranked
            if math.isfinite(s) and s >= required and str(w).casefold() not in target_cores_cf
        ]

        used: set[str] = set()
        for key in self.sorted_distractor_keys(out):
            val = out.get(key)
            if val is None:
                continue
            _, core, _ = strip_punctuation(str(val), self.puncts)
            core = str(core or "").strip()
            if core and core.casefold() not in target_cores_cf:
                used.add(core.casefold())

        def pick_replacement() -> Optional[str]:
            for cand in fallback_pool:
                cand_cf = str(cand).strip().casefold()
                if cand_cf not in used:
                    used.add(cand_cf)
                    return cand
            return None

        for key in self.sorted_distractor_keys(out):
            val = out.get(key)
            pre, dist_core, suf = ("", "", "")
            if val is not None:
                pre, dist_core, suf = strip_punctuation(str(val), self.puncts)
            dist_core = str(dist_core or "").strip()

            needs_replace = False
            if val is None:
                needs_replace = True
            elif not dist_core or dist_core.casefold() in target_cores_cf:
                needs_replace = True
            else:
                dist_score = cand_scores.get(dist_core)
                if dist_score is None:
                    try:
                        vals = []
                        for v in variants:
                            s = surprisal_cached(str(v.get("context_prefix", "")), dist_core)
                            if s is None:
                                raise RuntimeError("surprisal failed")
                            vals.append(float(s))
                        dist_score = float(mean(vals)) if vals else float("-inf")
                    except Exception:
                        dist_score = float("-inf")
                if dist_score is None or not math.isfinite(dist_score) or dist_score < required:
                    needs_replace = True

            if needs_replace:
                replacement = pick_replacement()
                if replacement is not None:
                    new_core = match_capitalization(str(replacement), ref_core)
                    out[key] = attach_punctuation(new_core, pre, suf)

        out["target_surprisal_avg"] = round(target_avg, 4)
        out["required_surprisal"] = round(required, 4)
        return out

    def _process_slot(
        self,
        *,
        item_id: str,
        word_index: int,
        variants: Sequence[Dict[str, Any]],
        repair: bool = True,
    ) -> Dict[str, Any]:
        cores = [str(v.get("target_core", "")).strip() for v in variants]
        source = self._compose_shared_source(cores)
        forced_lens = [int(v["forced_dummy_len"]) for v in variants if v.get("forced_dummy_len") is not None]

        if word_index == 0 or forced_lens:
            if forced_lens:
                dummy_len = int(round(mean(forced_lens)))
            else:
                lengths = [max(1, len(c)) for c in cores if c]
                dummy_len = int(round(mean(lengths))) if lengths else 1
            out = build_dummy_output(source=source, dummy_len=dummy_len, num_distractors=self.num_distractors)
        else:
            unique_cores = sorted({c for c in cores if c})
            if len(unique_cores) <= 1:
                rep = variants[0]
                raw = self.invoke_raw(
                    str(rep["sentence_prefix"]),
                    str(rep["target_core"]),
                    rep["candidates"],
                )
                out = self.parse_to_dict(raw, repair=repair)
                if not out:
                    out = build_dummy_output(
                        source=source,
                        dummy_len=max(1, len(str(rep["target_core"]))),
                        num_distractors=self.num_distractors,
                    )
                out = self.ensure_distractor_count(
                    out,
                    target_word=str(rep["target_word"]),
                    target_core=str(rep["target_core"]),
                    candidate_pool=rep["candidates"],
                    allow_candidate_fill=True,
                )
                out = self._enforce_avg_surprisal_threshold(
                    out,
                    variants=variants,
                    candidate_pool=rep["candidates"],
                )
            else:
                # Profile neighbor pool is shared; anchor selector on one random condition so {WORD}
                # matches the last token of {SENTENCE_PREFIX} (see maze templates).
                candidate_pool, avg_len, avg_rank = self._select_profile_candidates(variants)
                rep = random.choice(list(variants))
                raw = self.invoke_raw(
                    str(rep["sentence_prefix"]),
                    str(rep["target_core"]),
                    candidate_pool,
                )
                out = self.parse_to_dict(raw, repair=repair)
                if not out:
                    out = build_dummy_output(
                        source=source,
                        dummy_len=max(1, avg_len),
                        num_distractors=self.num_distractors,
                    )

                out = self.ensure_distractor_count(
                    out,
                    target_word=str(rep["target_word"]),
                    target_core=str(rep["target_core"]),
                    candidate_pool=candidate_pool,
                    allow_candidate_fill=True,
                )

                out = self._enforce_avg_surprisal_threshold(
                    out,
                    variants=variants,
                    candidate_pool=candidate_pool,
                )

                out["avg_target_length"] = int(avg_len)
                out["avg_target_rank"] = int(avg_rank)
                out["shared_across_conditions"] = True

        pfx, sfx = self._shared_punct(variants)
        if pfx is not None and sfx is not None:
            out = self.reattach_punct(out, pfx, sfx)

        out.update(
            item_id=item_id,
            word_index=word_index,
            condition_words={str(v["condition_id"]): str(v["target_word"]) for v in variants},
        )
        return out

    def run(
        self,
        controlled_items: Sequence[Dict[str, Any]],
        repair: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        run_start = time.perf_counter()
        total_items = len(controlled_items)
        total_slots = (
            sum(len(item["conditions"][0]["tokens"]) for item in controlled_items) if controlled_items else 0
        )
        print(f"[LLMAgent] Controlled mode: {total_items} items, {total_slots} item-word slots.")

        out_items: List[Dict[str, Any]] = []
        processed_slots = 0
        for item in controlled_items:
            if limit is not None and processed_slots >= limit:
                break
            item_id = str(item["item_id"])
            conditions = list(item["conditions"])
            sent_len = len(conditions[0]["tokens"])
            cond_ids = [str(c["condition_id"]) for c in conditions]

            words: List[Dict[str, Any]] = []
            for w_idx in range(sent_len):
                if limit is not None and processed_slots >= limit:
                    break
                token_metas = [self._token_metadata(str(cond["tokens"][w_idx])) for cond in conditions]
                need_candidate_pools = self._slot_needs_per_condition_candidate_pools(w_idx, token_metas)
                pool_by_token: Dict[str, tuple[str, str, str, tuple[str, ...], Optional[int]]] = {}

                variants: List[Dict[str, Any]] = []
                for cond, meta in zip(conditions, token_metas):
                    token = str(cond["tokens"][w_idx])
                    sentence_prefix = join_tokens(
                        cond["tokens"][: w_idx + 1],
                        join_with=self.token_joiner,
                        puncts=self.puncts,
                    )
                    context_prefix = join_tokens(
                        cond["tokens"][:w_idx],
                        join_with=self.token_joiner,
                        puncts=self.puncts,
                    )
                    if need_candidate_pools:
                        if token not in pool_by_token:
                            pfx, core, sfx, candidates, forced_len = self.get_candidates(token)
                            pool_by_token[token] = (pfx, core, sfx, tuple(candidates), forced_len)
                        pfx, core, sfx, candidates, forced_len = pool_by_token[token]
                        cand_list = list(candidates)
                    else:
                        pfx, core, sfx, forced_len = meta
                        cand_list = []
                    variants.append(
                        {
                            "condition_id": str(cond["condition_id"]),
                            "target_word": token,
                            "target_core": core,
                            "punct_prefix": pfx,
                            "punct_suffix": sfx,
                            "candidates": cand_list,
                            "forced_dummy_len": forced_len,
                            "sentence_prefix": sentence_prefix,
                            "context_prefix": context_prefix,
                            "word_index": int(w_idx),
                        }
                    )

                # normalize candidate containers to tuples for downstream consumers
                for v in variants:
                    v["candidates"] = tuple(v["candidates"])

                slot_out = self._process_slot(
                    item_id=item_id,
                    word_index=w_idx,
                    variants=variants,
                    repair=repair,
                )
                words.append(slot_out)
                processed_slots += 1

            out_items.append(
                {
                    "item_id": item_id,
                    "condition_ids": cond_ids,
                    "words": words,
                }
            )

        elapsed = time.perf_counter() - run_start
        print(f"[LLMAgent] Controlled timing summary: total={elapsed:.2f}s, processed_slots={processed_slots}")
        return out_items

