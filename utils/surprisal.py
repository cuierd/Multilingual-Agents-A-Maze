from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.components import attach_punctuation, strip_punctuation
from utils.distractor_utils import match_capitalization


@dataclass
class SurprisalThresholdConfig:
    enabled: bool = False
    min_abs: Optional[float] = None
    min_delta: float = 0.0
    absolute_threshold_only: bool = False


def torch_generation_surprisal_devices_compatible(gen_device: torch.device, requested: torch.device) -> bool:
    """True if scorer can run forwards on tensors placed on requested without moving shared weights."""
    if gen_device.type != requested.type:
        return False
    if requested.type == "cpu":
        return gen_device.type == "cpu"

    gi = gen_device.index
    ri = requested.index
    if gi is None and ri is None:
        return True
    gi = gi if gi is not None else 0
    ri = ri if ri is not None else 0
    return int(gi) == int(ri)


class SurprisalScorer:
    """Compute surprisal(word | prefix) in bits, summed over sub-tokens."""

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        *,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_id_display = model_id

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
            gen_device = next(self.model.parameters()).device
            tgt = torch.device(device or gen_device)
            if not torch_generation_surprisal_devices_compatible(gen_device, tgt):
                raise ValueError(
                    f"SurprisalScorer was given shared model on {gen_device}, but device={repr(device)} "
                    f"implies {tgt}. Omit tokenizer/model so a scorer copy can load on that device,"
                    " or align SURPRISAL_DEVICE / pipeline placement."
                )
            self.device = gen_device
            self.model.eval()
        elif tokenizer is None and model is None:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.tokenizer = self._load_tokenizer(model_id)
            self.model = self._load_model(model_id).to(self.device)
            self.model.eval()
        else:
            raise ValueError("SurprisalScorer: pass both tokenizer and model, or neither.")

        self.max_len = getattr(self.model.config, "n_positions", getattr(self.model.config, "max_position_embeddings", 2048))

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._prepend_bos = bool(
            self.tokenizer.bos_token_id is not None
            and (
                bool(getattr(self.tokenizer, "add_bos_token", False))
                or bool(getattr(self.model.config, "add_bos_token", False))
            )
        )

    @staticmethod
    def _load_tokenizer(model_id: str):
        last_error = None
        attempts = (
            {"use_fast": True, "local_files_only": True},
            {"use_fast": False, "local_files_only": True},
            {"use_fast": True},
            {"use_fast": False},
        )
        for kwargs in attempts:
            try:
                return AutoTokenizer.from_pretrained(model_id, **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to load tokenizer for surprisal model '{model_id}'.") from last_error

    @staticmethod
    def _load_model(model_id: str):
        last_error = None
        for kwargs in ({"local_files_only": True}, {}):
            try:
                return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            except Exception as exc:
                last_error = exc

        msg = f"Failed to load causal LM for surprisal model '{model_id}'."
        if last_error is not None:
            err_text = str(last_error)
            if "upgrade torch to at least v2.6" in err_text:
                msg += (
                    " The selected model appears to rely on PyTorch checkpoint loading (.bin), "
                    "but your environment's torch version is below 2.6."
                    "Fix options: (1) upgrade torch to >=2.6, or (2) choose a MODEL_ID "
                    "distributed as safetensors."
                )
            else:
                msg += f" Root cause: {err_text}"
        raise RuntimeError(msg) from last_error

    @staticmethod
    def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def word_surprisal(self, prefix: str, word: str, token_joiner: str = "") -> float:
        prefix_text = str(prefix or "")
        word_text = str(word or "")
        if not word_text:
            return 0.0

        join_text = str(token_joiner or "") if prefix_text else ""
        full_text = f"{prefix_text}{join_text}{word_text}"

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        split_idx = self._common_prefix_len(prefix_ids, full_ids)
        target_ids = full_ids[split_idx:]
        if not target_ids:
            return 0.0

        bos_ids = [self.tokenizer.bos_token_id] if self._prepend_bos else []
        allowed_ctx = max(0, self.max_len - len(bos_ids) - len(target_ids))
        context_ids = full_ids[:split_idx]
        if len(context_ids) > allowed_ctx:
            context_ids = context_ids[-allowed_ctx:]

        seq_ids = bos_ids + context_ids + target_ids
        target_start = len(bos_ids) + len(context_ids)
        if target_start <= 0:
            return float("inf")

        input_ids = torch.tensor([seq_ids], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            pred_ids = input_ids[:, 1:]
            token_logps = log_probs[:, :-1, :].gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)

            start_idx = target_start - 1
            end_idx = start_idx + len(target_ids)
            selected = token_logps[0, start_idx:end_idx]
            total_ln = -selected.sum().item()
            return float(total_ln / math.log(2))


class SurprisalThresholdFilter:
    """Apply surprisal-threshold constraints to Maze distractor outputs."""

    def __init__(
        self,
        config: SurprisalThresholdConfig,
        scorer: Optional[SurprisalScorer] = None,
    ) -> None:
        self.config = config
        self.scorer = scorer

    def _required_surprisal(self, target_surprisal: float) -> float:
        if self.config.absolute_threshold_only:
            return float(self.config.min_abs or 0.0)
        min_delta_rule = target_surprisal + float(self.config.min_delta or 0.0)
        if self.config.min_abs is None:
            return min_delta_rule
        return max(min_delta_rule, float(self.config.min_abs))

    def _score_word(self, context_prefix: str, word: str, token_joiner: str) -> Optional[float]:
        if not self.scorer:
            return None
        try:
            return self.scorer.word_surprisal(context_prefix, word, token_joiner=token_joiner)
        except Exception:
            return None

    def _score_candidates(
        self,
        context_prefix: str,
        candidates: Sequence[str],
        *,
        token_joiner: str = "",
    ) -> Dict[str, float]:
        if not self.scorer or not candidates:
            return {}
        out: Dict[str, float] = {}
        for cand in candidates:
            score = self._score_word(context_prefix, cand, token_joiner)
            if score is not None:
                out[cand] = score
        return out

    @staticmethod
    def _sorted_distractor_keys(out: Dict[str, Any]) -> Sequence[str]:
        keyed = []
        for key in out.keys():
            if not key.startswith("distractor"):
                continue
            suffix = key[len("distractor") :]
            if suffix.isdigit():
                keyed.append((int(suffix), key))
        keyed.sort(key=lambda x: x[0])
        return [k for _, k in keyed]

    def enforce_output_threshold(
        self,
        out: Dict[str, Any],
        *,
        context_prefix: str,
        target_core: str,
        candidate_pool: Sequence[str],
        puncts: Set[str],
        token_joiner: str = "",
    ) -> Dict[str, Any]:
        if not self.config.enabled or not self.scorer or not target_core:
            return out

        target_core = str(target_core).strip()
        target_cf = target_core.casefold()

        target_surprisal = self._score_word(context_prefix, target_core, token_joiner)
        if target_surprisal is None or not math.isfinite(target_surprisal):
            return out

        required = self._required_surprisal(target_surprisal)

        # Normalize candidate pool to punctuation-free, capitalization-matched cores.
        clean_pool: list[str] = []
        seen_pool: set[str] = set()

        for cand in candidate_pool:
            _, core, _ = strip_punctuation(str(cand), puncts)
            core = str(core or "").strip()
            if not core:
                continue

            core = match_capitalization(core, target_core)
            core_cf = core.casefold()

            if core_cf == target_cf or core_cf in seen_pool:
                continue

            seen_pool.add(core_cf)
            clean_pool.append(core)

        scored_pool = self._score_candidates(
            context_prefix,
            clean_pool,
            token_joiner=token_joiner,
        )

        ranked_pool = sorted(scored_pool.items(), key=lambda x: x[1], reverse=True)

        fallback_pool: list[str] = []
        for w, s in ranked_pool:
            if math.isfinite(s) and s >= required:
                fallback_pool.append(w)

        # Mark existing distractors as used by casefolded core.
        used: set[str] = set()
        for key in self._sorted_distractor_keys(out):
            value = out.get(key)
            if value is None:
                continue

            _, core, _ = strip_punctuation(str(value), puncts)
            core = str(core or "").strip()

            if core and core.casefold() != target_cf:
                used.add(core.casefold())

        def pick_replacement() -> Optional[str]:
            for cand in fallback_pool:
                cand_cf = str(cand).strip().casefold()
                if cand_cf not in used:
                    used.add(cand_cf)
                    return cand
            return None

        for key in self._sorted_distractor_keys(out):
            value = out.get(key)

            pre, dist_core, suf = ("", "", "")
            if value is not None:
                pre, dist_core, suf = strip_punctuation(str(value), puncts)

            dist_core = str(dist_core or "").strip()

            invalid = False

            if value is None:
                invalid = True
            elif not dist_core or dist_core.casefold() == target_cf:
                invalid = True
            else:
                dist_surp = self._score_word(context_prefix, dist_core, token_joiner)

                if dist_surp is None or not math.isfinite(dist_surp):
                    invalid = True
                elif dist_surp < required:
                    invalid = True

            if invalid:
                if dist_core:
                    used.discard(dist_core.casefold())

                replacement = pick_replacement()

                if replacement is not None:
                    new_core = match_capitalization(str(replacement), target_core)
                    out[key] = attach_punctuation(new_core, pre, suf)

        if self.config.min_abs is not None:
            out["min_abs_threshold"] = float(self.config.min_abs)

        out["min_delta_threshold"] = float(self.config.min_delta or 0.0)
        out["absolute_threshold_only"] = bool(self.config.absolute_threshold_only)
        out["target_surprisal"] = round(float(target_surprisal), 4)
        out["required_surprisal"] = round(float(required), 4)

        return out
