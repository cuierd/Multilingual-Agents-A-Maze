from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from json_repair import repair_json
from langchain_huggingface import HuggingFacePipeline

from utils.candidate_proposal import (
    LexiconCandidateCache,
    apply_model_proposed_candidates,
    extract_hf_generated_text,
    parse_distractor_pool_lenient,
    resolve_surprisal_model_id,
    lexicon_candidates,
)
from utils.components import (
    Lexicon,
    get_punctuation,
    join_tokens,
    load_config,
    read_sentences_input,
    strip_punctuation,
    token_joiner_for_language,
)
from utils.controlled_mode import ControlledModeService, read_controlled_input as read_controlled_input_file
from utils.distractor_utils import (
    ensure_distractor_count,
    extract_forced_target,
    reattach_punctuation_to_output,
    sorted_distractor_keys,
)
from utils.runtime_config import build_llm_runtime_config, validate_controlled_experiment_lexicon
from utils.surprisal import (
    SurprisalScorer,
    SurprisalThresholdConfig,
    SurprisalThresholdFilter,
    torch_generation_surprisal_devices_compatible,
)


@dataclass
class AgentConfig:
    """Configuration for HF-based maze generation."""

    model_id: str
    lexicon_path: Optional[str]
    punctuations: Sequence[str]
    # Empty / None falls back to model_id so one model does both stages by default.
    proposal_model_id: Optional[str] = None
    selection_model_id: Optional[str] = None
    use_lexicon_candidates: bool = True
    generator_prompt: Any = None
    # generation params
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    return_full_text: bool = False
    # lexicon params
    min_candidates: int = 10
    max_candidates: int = 20
    num_distractors: int = 3
    num_workers: int = 1
    # surprisal threshold params
    apply_surprisal_threshold: bool = False
    min_abs: Optional[float] = None
    min_delta: float = 0.0
    absolute_threshold_only: bool = False
    surprisal_device: Optional[str] = None
    # token joining strategy (e.g., "" for CJK char tokens, " " for space-separated)
    token_joiner: str = ""


class LLMAgent:
    """
    For each sentence (list of tokens), iterate i=1..len-1:
      - prefix_sentence = cfg.token_joiner.join(tokens[:i+1])  (ends at target word)
      - target_word = tokens[i]
      - candidates: lexicon neighborhoods *or* (two-stage HF) Stage-A generated pool
      - Stage-B HF (``MazeLLMPrompt``) chooses distractors from candidates
      - optional: enforce surprisal thresholds over chosen distractors

    Single-shot operation was lexicon -> one HF selector call; when
    ``use_lexicon_candidates`` is False, Stage-A proposes and Stage-B selects (different HF
    models allowed via ``proposal_model_id`` / ``selection_model_id``).
    """

    #: Stage-A HF calls per job (max rounds of candidate proposal)
    _GENERATOR_POOL_MAX_ROUNDS: int = 2

    def __init__(
        self,
        prompt: Any,
        cfg: AgentConfig,
        llm_selector: Optional[HuggingFacePipeline] = None,
        llm_proposal: Optional[HuggingFacePipeline] = None,
    ) -> None:
        self.prompt = prompt
        self.cfg = cfg
        if not 1 <= int(self.cfg.num_distractors) <= 10:
            raise ValueError("num_distractors must be in [1, 10].")
        if int(self.cfg.num_workers) < 1:
            raise ValueError("num_workers must be >= 1.")

        self.proposal_mid = str(self.cfg.proposal_model_id or self.cfg.model_id).strip()
        self.selection_mid = str(self.cfg.selection_model_id or self.cfg.model_id).strip()

        lp = Path(self.cfg.lexicon_path) if self.cfg.lexicon_path else None
        if bool(self.cfg.use_lexicon_candidates):
            if lp is None or not str(lp).strip() or not lp.exists():
                raise ValueError(
                    "use_lexicon_candidates=True requires an existing LEXICON_PATH (or resolved default lexicon file)."
                )
            self.lexicon: Optional[Lexicon] = Lexicon(str(lp))
            self._lexicon_cache = LexiconCandidateCache(self.lexicon)
        else:
            if self.cfg.generator_prompt is None:
                raise ValueError("use_lexicon_candidates=False requires generator_prompt (DistractorGeneratorPrompt).")
            self.lexicon = None
            self._lexicon_cache = None

        self.generator_prompt = self.cfg.generator_prompt
        self.puncts = set(self.cfg.punctuations)

        build_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "do_sample": self.cfg.do_sample,
            "return_full_text": self.cfg.return_full_text,
        }
        self.llm_selector = llm_selector or HuggingFacePipeline.from_model_id(
            model_id=self.selection_mid,
            task="text-generation",
            pipeline_kwargs=dict(build_kwargs),
        )
        if self.proposal_mid == self.selection_mid:
            self.llm_proposal = llm_proposal or self.llm_selector
        else:
            self.llm_proposal = llm_proposal or HuggingFacePipeline.from_model_id(
                model_id=self.proposal_mid,
                task="text-generation",
                pipeline_kwargs=dict(build_kwargs),
            )
        self.llm = self.llm_selector
        self._configure_generation_padding(self.llm_selector)
        if self.llm_proposal is not self.llm_selector:
            self._configure_generation_padding(self.llm_proposal)

        self._threshold_filter = SurprisalThresholdFilter(
            config=SurprisalThresholdConfig(enabled=False)
        )
        if self.cfg.apply_surprisal_threshold:
            scorer_mid = resolve_surprisal_model_id(
                primary_model_id=self.cfg.model_id,
                selection_model_id=self.selection_mid,
            )
            pipe = getattr(self.llm_selector, "pipeline", None)
            pipe_tok = getattr(pipe, "tokenizer", None) if pipe is not None else None
            pipe_mod = getattr(pipe, "model", None) if pipe is not None else None
            scorer_device = torch.device(
                self.cfg.surprisal_device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            can_share_checkpoint = (
                pipe_tok is not None
                and pipe_mod is not None
                and str(scorer_mid).strip() == str(self.selection_mid).strip()
            )
            gen_device = (
                next(pipe_mod.parameters()).device if pipe_mod is not None else None
            )
            can_share_weights = (
                bool(can_share_checkpoint)
                and gen_device is not None
                and torch_generation_surprisal_devices_compatible(gen_device, scorer_device)
            )
            if can_share_checkpoint and not can_share_weights and gen_device is not None:
                print(
                    f"[LLMAgent] Surprisal model matches selector ({scorer_mid}), but SURPRISAL_DEVICE "
                    f"({repr(self.cfg.surprisal_device)}) implies {scorer_device} vs generator on {gen_device}; "
                    "loading a second causal LM copy for scoring."
                )
            if can_share_weights:
                print(
                    f"[LLMAgent] SurprisalScorer reuses selector causal LM ({scorer_mid}); "
                    "no second from_pretrained download/load."
                )
                scorer = SurprisalScorer(
                    scorer_mid,
                    tokenizer=pipe_tok,
                    model=pipe_mod,
                )
            else:
                scorer = SurprisalScorer(
                    scorer_mid,
                    device=self.cfg.surprisal_device,
                )
            self._threshold_filter = SurprisalThresholdFilter(
                config=SurprisalThresholdConfig(
                    enabled=True,
                    min_abs=self.cfg.min_abs,
                    min_delta=self.cfg.min_delta,
                    absolute_threshold_only=self.cfg.absolute_threshold_only,
                ),
                scorer=scorer,
            )

    def _configure_generation_padding(self, llm: HuggingFacePipeline) -> None:
        """Avoid right-padding warnings for decoder-only batched generation."""
        pipe = getattr(llm, "pipeline", None)
        if pipe is None:
            return
        tokenizer = getattr(pipe, "tokenizer", None)
        model = getattr(pipe, "model", None)
        if tokenizer is None:
            return

        model_cfg = getattr(model, "config", None)
        is_decoder_only = not bool(getattr(model_cfg, "is_encoder_decoder", False))
        if not is_decoder_only:
            return

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        if model_cfg is not None and getattr(model_cfg, "pad_token_id", None) is None:
            model_cfg.pad_token_id = tokenizer.pad_token_id

    # ---------- helpers ----------
    def _non_pad_proposal_count(self, raw_word_list: List[str], core: str) -> int:
        """How many finalized pool slots are not ``X`` padding for ``core`` (after apply + pad rules)."""
        pad_token = "X" * max(1, len(str(core)))
        cand_list = apply_model_proposed_candidates(
            raw_word_list,
            str(core),
            min_candidates=self.cfg.min_candidates,
            max_candidates=self.cfg.max_candidates,
        )
        return sum(1 for c in cand_list if str(c).strip() and str(c).strip() != pad_token)

    def _merge_hf_proposal_rounds_for_jobs(
        self,
        jobs: List[Dict[str, Any]],
        *,
        sentence_prefix_key: str = "sentence_prefix",
        target_core_key: str = "target_core",
    ) -> None:
        """
        Mutates each job: sets ``candidates`` to a tuple from merged Stage-A outputs.
        Batches HF calls per round; follow-up rounds only for jobs still below ``min_candidates`` real words.
        """
        if not jobs or self.generator_prompt is None:
            return
        merged: List[List[str]] = [[] for _ in jobs]
        for attempt in range(max(1, int(self._GENERATOR_POOL_MAX_ROUNDS))):
            active = [
                i
                for i in range(len(jobs))
                if self._non_pad_proposal_count(merged[i], str(jobs[i][target_core_key]))
                < int(self.cfg.min_candidates)
            ]
            if not active:
                break
            prompts = [
                self.generator_prompt.render_text(  # type: ignore[union-attr]
                    sentence_prefix=str(jobs[i][sentence_prefix_key]),
                    word=str(jobs[i][target_core_key]),
                )
                for i in active
            ]
            if attempt == 0:
                print(
                    f"[LLMAgent] Candidate proposal batch: {len(jobs)} jobs "
                    f"(proposal model={self.proposal_mid})."
                )
            else:
                print(
                    f"[LLMAgent] Candidate proposal round {attempt + 1}: {len(active)} jobs "
                    f"(still below min non-padding count)."
                )
            raws = self._invoke_raw_batch(prompts, hf=self.llm_proposal)
            if len(raws) != len(active):
                raws = [
                    extract_hf_generated_text(self.llm_proposal.invoke(p)) for p in prompts
                ]
            for idx, raw in zip(active, raws):
                merged[idx].extend(parse_distractor_pool_lenient(raw))

        for job, words in zip(jobs, merged):
            job["candidates"] = tuple(
                apply_model_proposed_candidates(
                    words,
                    str(job[target_core_key]),
                    min_candidates=self.cfg.min_candidates,
                    max_candidates=self.cfg.max_candidates,
                )
            )

    def _fill_pending_candidate_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """Batch Stage-A HF for jobs flagged with pending_candidate_proposal."""
        pending = [j for j in jobs if j.get("pending_candidate_proposal")]
        if not pending:
            return
        self._merge_hf_proposal_rounds_for_jobs(pending)
        for job in pending:
            job["pending_candidate_proposal"] = False

    def _reattach_punct(self, out: Dict[str, Any], prefix: str, suffix: str) -> Dict[str, Any]:
        return reattach_punctuation_to_output(out, prefix=prefix, suffix=suffix)

    @staticmethod
    def _sorted_distractor_keys(out: Dict[str, Any]) -> List[str]:
        return sorted_distractor_keys(out)

    def _ensure_distractor_count(
        self,
        out: Dict[str, Any],
        *,
        target_word: str,
        target_core: str,
        candidate_pool: Sequence[str],
        allow_candidate_fill: bool = True,
    ) -> Dict[str, Any]:
        return ensure_distractor_count(
            out,
            num_distractors=self.cfg.num_distractors,
            target_word=target_word,
            target_core=target_core,
            candidate_pool=candidate_pool,
            puncts=self.puncts,
            allow_candidate_fill=allow_candidate_fill,
        )

    @staticmethod
    def _extract_forced_target(core: str) -> Optional[str]:
        return extract_forced_target(core)

    # ---------- candidate generation ----------

    def get_candidates(self, word: str) -> tuple[str, str, str, tuple[str, ...], Optional[int]]:
        prefix, core, suffix = strip_punctuation(word, self.puncts)
        if not core:
            return prefix, core, suffix, ("X" * len(word),), None
        forced_target = self._extract_forced_target(core)
        if forced_target is not None:
            forced_len = max(1, len(forced_target))
            return prefix, forced_target, suffix, tuple(), forced_len

        if self.cfg.use_lexicon_candidates:
            assert self._lexicon_cache is not None
            cands = lexicon_candidates(
                self._lexicon_cache,
                core,
                min_candidates=self.cfg.min_candidates,
                max_candidates=self.cfg.max_candidates,
            )
            return prefix, core, suffix, tuple(cands), None

        # Filled via Stage-A HF batch in naturalistic runs without lexicon.
        return prefix, core, suffix, tuple(), None

    # ---------- LLM + parsing ----------

    def _invoke_raw(self, sentence_prefix: str, target_word: str, candidates: Sequence[str]) -> str:
        prompt_text = self.prompt.render_text(
            sentence_prefix=sentence_prefix,
            word=target_word,
            candidates=candidates,
        )
        resp = self.llm_selector.invoke(prompt_text)
        return extract_hf_generated_text(resp)

    def _invoke_raw_batch(
        self,
        prompts: Sequence[str],
        *,
        hf: HuggingFacePipeline,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """Batch invoke HF pipeline; prefer dataset-style streaming on GPU."""
        prompt_list = list(prompts)
        total = len(prompt_list)
        if total == 0:
            return []

        def _report(done: int) -> None:
            if progress_callback is not None:
                progress_callback(done, total)

        pipe = getattr(hf, "pipeline", None)
        if pipe is not None:
            try:
                from transformers.pipelines.pt_utils import KeyDataset

                class _PromptDataset:
                    def __init__(self, rows: Sequence[str]) -> None:
                        self.rows = rows

                    def __len__(self) -> int:
                        return len(self.rows)

                    def __getitem__(self, idx: int) -> Dict[str, str]:
                        return {"text": self.rows[idx]}

                dataset = _PromptDataset(prompt_list)
                streamed = pipe(
                    KeyDataset(dataset, "text"),
                    batch_size=max(1, int(self.cfg.num_workers)),
                )
                out: List[str] = []
                for i, item in enumerate(streamed, start=1):
                    out.append(extract_hf_generated_text(item))
                    _report(i)
                return out
            except Exception:
                pass

            try:
                batch_out = pipe(
                    prompt_list,
                    batch_size=max(1, int(self.cfg.num_workers)),
                )
                out = [extract_hf_generated_text(item) for item in batch_out]
                for i in range(1, len(out) + 1):
                    _report(i)
                return out
            except Exception:
                pass

        out: List[str] = []
        for i, p in enumerate(prompt_list, start=1):
            out.append(extract_hf_generated_text(hf.invoke(p)))
            _report(i)
        return out

    def _parse_to_dict(self, raw: str, repair: bool = True) -> Dict[str, Any]:
        try:
            obj = self.prompt.parser.parse(raw)
            return obj.model_dump()
        except Exception:
            if not repair:
                raise
        try:
            fixed = repair_json(raw)
            data = json.loads(fixed)
        except Exception:
            return {}
        if isinstance(data, list):
            for x in reversed(data):
                if isinstance(x, dict):
                    data = x
                    break
        if not isinstance(data, dict):
            return {}
        data.setdefault("source", "")
        data.setdefault("distractor1", "X")
        data.setdefault("distractor2", "X")
        data.setdefault("distractor3", None)
        try:
            obj = self.prompt.parser.pydantic_object.model_validate(data)
            return obj.model_dump()
        except Exception:
            return {}


    # ---------- public API ----------

    def iter_jobs(self, input_data: Sequence[Sequence[str]]) -> Iterable[Dict[str, Any]]:
        for s_idx, tokens in enumerate(input_data):
            for i in range(len(tokens)):
                target = tokens[i]
                sentence_prefix = join_tokens(tokens[: i + 1], join_with=self.cfg.token_joiner, puncts=self.puncts)
                context_prefix = join_tokens(tokens[:i], join_with=self.cfg.token_joiner, puncts=self.puncts)
                pfx, core, sfx, candidates, forced_len = self.get_candidates(target)
                pending_candidate_proposal = (
                    not bool(self.cfg.use_lexicon_candidates)
                    and i > 0
                    and forced_len is None
                )
                yield {
                    "sentence_index": s_idx,
                    "word_index": i,
                    "sentence_prefix": sentence_prefix,
                    "context_prefix": context_prefix,
                    "target_word": target,
                    "target_core": core,
                    "punct_prefix": pfx,
                    "punct_suffix": sfx,
                    "candidates": candidates,
                    "forced_dummy_len": forced_len,
                    "pending_candidate_proposal": pending_candidate_proposal,
                }

    def read_controlled_input(self, path: Union[str, Path], split_on: Optional[str]) -> List[Dict[str, Any]]:
        return read_controlled_input_file(path, split_on)

    def _process_job(self, job: Dict[str, Any], *, repair: bool = True) -> Dict[str, Any]:
        forced_len = job.get("forced_dummy_len")
        if job["word_index"] == 0 or forced_len is not None:
            dummy = "X" * int(forced_len if forced_len is not None else len(job["target_word"]))
            out = {"source": job["target_core"], "distractor1": dummy}
            out = self._ensure_distractor_count(
                out,
                target_word=job["target_word"],
                target_core=job["target_core"],
                candidate_pool=job["candidates"],
                allow_candidate_fill=False,
            )
            if forced_len is not None:
                out = self._reattach_punct(out, job["punct_prefix"], job["punct_suffix"])
        else:
            raw = self._invoke_raw(job["sentence_prefix"], job["target_core"], job["candidates"])
            out = self._parse_to_dict(raw, repair=repair)
            # Fallback if parsing totally failed
            if not out:
                dummy = "X" * len(job["target_word"])
                out = {
                    "source": job["target_core"],
                    "distractor1": dummy,
                    "distractor2": dummy,
                    "distractor3": None,
                }
            out = self._ensure_distractor_count(
                out,
                target_word=job["target_word"],
                target_core=job["target_core"],
                candidate_pool=job["candidates"],
                allow_candidate_fill=True,
            )
            out = self._threshold_filter.enforce_output_threshold(
                out=out,
                context_prefix=job["context_prefix"],
                target_core=job["target_core"],
                candidate_pool=job["candidates"],
                puncts=self.puncts,
                token_joiner=self.cfg.token_joiner,
            )
            out = self._reattach_punct(out, job["punct_prefix"], job["punct_suffix"])

        # attach metadata (useful for reconstructing maze items)
        out.update(
            sentence_index=job["sentence_index"],
            word_index=job["word_index"],
            sentence_prefix=job["sentence_prefix"],
            target_word=job["target_word"],
        )
        return out

    def _process_job_with_raw(self, job: Dict[str, Any], raw: str, *, repair: bool = True) -> Dict[str, Any]:
        out = self._parse_to_dict(raw, repair=repair)
        if not out:
            dummy = "X" * len(job["target_word"])
            out = {
                "source": job["target_core"],
                "distractor1": dummy,
                "distractor2": dummy,
                "distractor3": None,
            }
        out = self._ensure_distractor_count(
            out,
            target_word=job["target_word"],
            target_core=job["target_core"],
            candidate_pool=job["candidates"],
            allow_candidate_fill=True,
        )
        out = self._threshold_filter.enforce_output_threshold(
            out=out,
            context_prefix=job["context_prefix"],
            target_core=job["target_core"],
            candidate_pool=job["candidates"],
            puncts=self.puncts,
            token_joiner=self.cfg.token_joiner,
        )
        out = self._reattach_punct(out, job["punct_prefix"], job["punct_suffix"])
        out.update(
            sentence_index=job["sentence_index"],
            word_index=job["word_index"],
            sentence_prefix=job["sentence_prefix"],
            target_word=job["target_word"],
        )
        return out

    def run(
        self,
        input_data: Sequence[Sequence[str]],
        repair: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run over input_data and return a sentence-grouped list of JSON dict outputs."""
        run_start = time.perf_counter()
        jobs = list(self.iter_jobs(input_data))
        if limit is not None:
            jobs = jobs[:limit]
        if not jobs:
            return []

        self._fill_pending_candidate_jobs(jobs)

        total_sentences = len(input_data)
        print(f"[LLMAgent] Processing {len(jobs)} word-jobs from {total_sentences} sentences...")
        print(f"[LLMAgent] Generation batch size: {self.cfg.num_workers}")

        outputs: List[Dict[str, Any]] = []
        llm_jobs: List[Dict[str, Any]] = []
        llm_prompts: List[str] = []
        total_llm_seconds = 0.0

        for job in jobs:
            if job["word_index"] == 0 or job.get("forced_dummy_len") is not None:
                outputs.append(self._process_job(job, repair=repair))
            else:
                llm_jobs.append(job)
                llm_prompts.append(
                    self.prompt.render_text(
                        sentence_prefix=job["sentence_prefix"],
                        word=job["target_core"],
                        candidates=job["candidates"],
                    )
                )

        if llm_jobs:
            batch_size = max(1, int(self.cfg.num_workers))
            total_llm_jobs = len(llm_jobs)
            print(
                f"[LLMAgent] LLM generation jobs: {total_llm_jobs} "
                f"(dataset-style, batch_size={batch_size})"
            )

            llm_start = time.perf_counter()
            report_every = max(1, total_llm_jobs // 20)

            def _progress(done: int, total: int) -> None:
                if done == 1 or done == total or done % report_every == 0:
                    elapsed = time.perf_counter() - llm_start
                    avg = elapsed / max(1, done)
                    eta = avg * (total - done)
                    print(
                        f"[LLMAgent] Progress: {done}/{total} LLM jobs done, "
                        f"elapsed={elapsed:.2f}s, avg={avg:.2f}s/job, eta={eta:.2f}s"
                    )

            raw_outputs = self._invoke_raw_batch(llm_prompts, hf=self.llm_selector, progress_callback=_progress)
            if len(raw_outputs) != len(llm_jobs):
                raw_outputs = [
                    self._invoke_raw(j["sentence_prefix"], j["target_core"], j["candidates"])
                    for j in llm_jobs
                ]
            total_llm_seconds = time.perf_counter() - llm_start

            for job, raw in zip(llm_jobs, raw_outputs):
                outputs.append(self._process_job_with_raw(job, raw, repair=repair))

        outputs.sort(key=lambda x: (x["sentence_index"], x["word_index"]))
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for out in outputs:
            grouped.setdefault(int(out["sentence_index"]), []).append(out)
        run_seconds = time.perf_counter() - run_start
        if llm_jobs:
            print(
                f"[LLMAgent] Timing summary: total={run_seconds:.2f}s, "
                f"llm_calls={total_llm_seconds:.2f}s, "
                f"avg_llm_call={total_llm_seconds / max(1, len(llm_jobs)):.2f}s"
            )
        else:
            print(f"[LLMAgent] Timing summary: total={run_seconds:.2f}s")
        return [{"sentence_index": s_idx, "words": grouped[s_idx]} for s_idx in sorted(grouped.keys())]

    def run_controlled_experiment(
        self,
        controlled_items: Sequence[Dict[str, Any]],
        repair: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Controlled mode:
        - one shared distractor set per (item_id, word_index)
        - all conditions of the item reuse this shared set
        """
        validate_controlled_experiment_lexicon(self.cfg.use_lexicon_candidates)
        service = ControlledModeService(
            lexicon=self.lexicon,
            puncts=self.puncts,
            num_distractors=self.cfg.num_distractors,
            token_joiner=self.cfg.token_joiner,
            min_candidates=self.cfg.min_candidates,
            max_candidates=self.cfg.max_candidates,
            apply_surprisal_threshold=self.cfg.apply_surprisal_threshold,
            threshold_filter=self._threshold_filter,
            get_candidates=self.get_candidates,
            invoke_raw=self._invoke_raw,
            parse_to_dict=self._parse_to_dict,
            ensure_distractor_count=self._ensure_distractor_count,
            reattach_punct=self._reattach_punct,
            sorted_distractor_keys=self._sorted_distractor_keys,
        )
        return service.run(controlled_items=controlled_items, repair=repair, limit=limit)

    @staticmethod
    def save_jsonl(records: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
        """Machine-friendly JSONL (one object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def save_pretty_json(records: Sequence[Dict[str, Any]], path: Union[str, Path]) -> None:
        """Human-friendly JSON (pretty-printed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

# -------------------------
# minimal usage example
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Maze distractor generation (lexicon + LLM selector).")
    parser.add_argument("--config-path", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--language-code", default=None, help="Override LANGUAGE_CODE from config.")
    parser.add_argument("--min-abs", type=float, default=None, help="Minimum absolute distractor surprisal.")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum surprisal delta over target.")
    parser.add_argument(
        "--absolute-threshold-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only min_abs as hard threshold (ignore target + min_delta).",
    )
    parser.add_argument("--surprisal-device", default=None, help='Device for surprisal scorer, e.g. "cuda" or "cpu".')
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of word-jobs to run.")
    parser.add_argument("--output-path", default=None, help="Optional explicit output JSON path.")
    parser.add_argument("--model-id", default=None, help="Override MODEL_ID from config.")
    parser.add_argument("--input-data-path", default=None, help="Optional explicit input data path override.")
    parser.add_argument("--lexicon-path", default=None, help="Optional explicit lexicon path override.")
    parser.add_argument("--template-dir", default=None, help="Optional explicit template directory override.")
    parser.add_argument(
        "--processing-mode",
        default=None,
        choices=["naturalistic_reading", "controlled_experiment"],
        help="Processing mode: naturalistic_reading or controlled_experiment.",
    )
    parser.add_argument("--word-separator", default=None, help="Override WORD_SEPARATOR from config.")
    parser.add_argument("--num-distractors", type=int, default=None, help="Number of distractors per word (1-10).")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel workers for word-level processing (>=1).")
    parser.add_argument(
        "--proposal-model-id",
        default=None,
        help="Stage-A Hugging Face model id (candidate proposal). Defaults to MODEL_ID.",
    )
    parser.add_argument(
        "--selection-model-id",
        default=None,
        help="Stage-B Hugging Face model id (distractor selection). Defaults to MODEL_ID.",
    )
    parser.add_argument(
        "--use-lexicon-candidates/--no-use-lexicon-candidates",
        dest="use_lexicon_candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If enabled (default from config), read candidate neighborhoods from LEXICON_PATH. "
        "If disabled, Stage-A proposes candidates from LM output.",
    )
    args = parser.parse_args()

    from utils.maze_prompt import DistractorGeneratorPrompt, MazeLLMPrompt

    yaml_cfg = load_config(args.config_path)
    override_dict: Dict[str, Any] = {
        k: v
        for k, v in {
            "AGENT_TYPE": "llm",
            "LANGUAGE_CODE": args.language_code,
            "MODEL_ID": args.model_id,
            "INPUT_DATA_PATH": args.input_data_path,
            "LEXICON_PATH": args.lexicon_path,
            "TEMPLATE_DIR": args.template_dir,
            "OUTPUT_PATH": args.output_path,
            "PROCESSING_MODE": args.processing_mode,
            "WORD_SEPARATOR": args.word_separator,
            "NUM_DISTRACTORS": args.num_distractors,
            "NUM_WORKERS": args.num_workers,
            "SURPRISAL_MIN_ABS": args.min_abs,
            "SURPRISAL_MIN_DELTA": args.min_delta,
            "SURPRISAL_ABSOLUTE_THRESHOLD_ONLY": args.absolute_threshold_only,
            "SURPRISAL_DEVICE": args.surprisal_device,
            "PROPOSAL_MODEL_ID": args.proposal_model_id,
            "SELECTION_MODEL_ID": args.selection_model_id,
        }.items()
        if v is not None
    }
    if args.use_lexicon_candidates is not None:
        override_dict["USE_LEXICON_CANDIDATES"] = bool(args.use_lexicon_candidates)

    runtime_cfg = build_llm_runtime_config(
        yaml_cfg=yaml_cfg,
        overrides=override_dict,
    )

    language_code = runtime_cfg.language_code
    processing_mode = runtime_cfg.processing_mode
    main_path = Path(runtime_cfg.template_dir)
    word_separator = runtime_cfg.word_separator
    puncts = get_punctuation(language_code)
    if not main_path.exists():
        raise ValueError(f"Template directory not found for LANGUAGE_CODE='{language_code}': {main_path}")
    if runtime_cfg.use_lexicon_candidates and not Path(runtime_cfg.lexicon_path).exists():
        raise ValueError(f"Lexicon file not found: {runtime_cfg.lexicon_path}")
    if not Path(runtime_cfg.input_data_path).exists():
        raise ValueError(f"Input data file not found: {runtime_cfg.input_data_path}")

    maze_prompt = MazeLLMPrompt(
        path_to_user_template=main_path / "base.txt",
        path_to_extension_template=main_path / "extension.txt",
        path_to_system_template=main_path / "system.txt",
    )
    generator_prompt: Optional[Any] = None
    if not runtime_cfg.use_lexicon_candidates:
        generator_prompt = DistractorGeneratorPrompt(
            path_to_user_template=main_path / "distractor_gen_base.txt",
            path_to_extension_template=main_path / "distractor_gen_extension.txt",
            path_to_system_template=main_path / "system.txt",
        )

    primary_model_id = runtime_cfg.model_id

    def _explicit_stage_model(resolved_stage_id: str) -> Optional[str]:
        resolved_stage_id = str(resolved_stage_id).strip()
        return resolved_stage_id if resolved_stage_id != primary_model_id else None

    agent_cfg = AgentConfig(
        model_id=primary_model_id,
        proposal_model_id=_explicit_stage_model(runtime_cfg.proposal_model_id),
        selection_model_id=_explicit_stage_model(runtime_cfg.selection_model_id),
        lexicon_path=runtime_cfg.lexicon_path if runtime_cfg.use_lexicon_candidates else None,
        punctuations=puncts,
        num_distractors=runtime_cfg.num_distractors,
        num_workers=runtime_cfg.num_workers,
        apply_surprisal_threshold=runtime_cfg.apply_surprisal_threshold,
        min_abs=runtime_cfg.min_abs,
        min_delta=runtime_cfg.min_delta,
        absolute_threshold_only=runtime_cfg.absolute_threshold_only,
        surprisal_device=runtime_cfg.surprisal_device,
        token_joiner=token_joiner_for_language(language_code, word_separator=word_separator),
        use_lexicon_candidates=runtime_cfg.use_lexicon_candidates,
        generator_prompt=generator_prompt,
    )

    agent = LLMAgent(prompt=maze_prompt, cfg=agent_cfg)

    main_start = time.perf_counter()
    if processing_mode == "controlled_experiment":
        controlled_items = agent.read_controlled_input(
            runtime_cfg.input_data_path,
            split_on=word_separator,
        )
        outputs = agent.run_controlled_experiment(controlled_items, limit=args.limit)
    else:
        input_data = read_sentences_input(runtime_cfg.input_data_path, split_on=word_separator)
        outputs = agent.run(input_data, limit=args.limit)
    generation_elapsed = time.perf_counter() - main_start
    print(
        f"[LLMAgent] Generation finished in {generation_elapsed:.2f}s "
        f"(mode={processing_mode}, records={len(outputs)})."
    )
    output_path = args.output_path or runtime_cfg.output_path
    agent.save_pretty_json(outputs, output_path)
    total_elapsed = time.perf_counter() - main_start
    print(f"Saved output to: {output_path}")
    print(f"[LLMAgent] Total elapsed (generation + save): {total_elapsed:.2f}s")
