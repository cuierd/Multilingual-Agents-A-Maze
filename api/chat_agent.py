from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from json_repair import repair_json
from langchain_openai import ChatOpenAI

from utils.components import (
    Lexicon,
    get_punctuation,
    join_tokens,
    load_config,
    read_sentences_input,
    strip_punctuation,
    token_joiner_for_language,
)
from utils.candidate_proposal import (
    LexiconCandidateCache,
    apply_model_proposed_candidates,
    invoke_openai_generator,
    lexicon_candidates,
    resolve_surprisal_model_id,
)
from utils.surprisal import SurprisalScorer, SurprisalThresholdConfig, SurprisalThresholdFilter
from utils.controlled_mode import ControlledModeService, read_controlled_input as read_controlled_input_file
from utils.distractor_utils import (
    build_dummy_output,
    ensure_distractor_count,
    extract_forced_target,
    reattach_punctuation_to_output,
    sorted_distractor_keys,
)
from utils.maze_prompt import DistractorGeneratorPrompt, MazeChatPrompt
from utils.runtime_config import resolve_runtime_paths, validate_controlled_experiment_lexicon


@dataclass
class ChatAgentConfig:
    lexicon_path: Optional[str] = None
    language_code: Optional[str] = None
    punctuations: Sequence[str] = ()
    min_candidates: int = 10
    max_candidates: int = 20
    num_distractors: int = 3
    token_joiner: str = ""
    lexicon_mode: bool = False
    apply_surprisal_threshold: bool = False
    min_abs: Optional[float] = None
    min_delta: float = 0.0
    absolute_threshold_only: bool = False
    surprisal_device: Optional[str] = None
    model_id: str = "gpt-4o-mini"
    # Falls back to model_id when empty / None so one OpenAI deployment can handle both stages.
    proposal_model_id: Optional[str] = None
    selection_model_id: Optional[str] = None
    gen_temperature: float = 0.7
    sel_temperature: float = 0.2


class ChatAgent:
    #: Extra Stage-A calls when the model returns too few unique words (merged pools are de-duped
    #: in ``apply_model_proposed_candidates`` before placeholder padding).
    _GENERATOR_POOL_MAX_ROUNDS: int = 2

    def __init__(
        self,
        cfg: ChatAgentConfig,
        selector_prompt: Any,
        generator_prompt: Optional[Any] = None,
        *,
        lexicon_mode: Optional[bool] = None,
        chat_gen: Optional[ChatOpenAI] = None,
        chat_sel: Optional[ChatOpenAI] = None,
    ) -> None:
        self.cfg = cfg
        self.selector_prompt = selector_prompt
        self.generator_prompt = generator_prompt
        if not 1 <= int(self.cfg.num_distractors) <= 10:
            raise ValueError("num_distractors must be in [1, 10].")
        requested_lexicon_mode = bool(self.cfg.lexicon_mode) if lexicon_mode is None else bool(lexicon_mode)
        self.use_lexicon = False
        self.puncts = set(self.cfg.punctuations)
        proposal_mid = str(self.cfg.proposal_model_id or self.cfg.model_id).strip()
        selection_mid = str(self.cfg.selection_model_id or self.cfg.model_id).strip()
        self.chat_gen = chat_gen or ChatOpenAI(model=proposal_mid, temperature=self.cfg.gen_temperature)
        self.chat_sel = chat_sel or ChatOpenAI(model=selection_mid, temperature=self.cfg.sel_temperature)

        self.lexicon_obj: Optional[Lexicon] = None
        self._lexicon_cache: Optional[LexiconCandidateCache] = None
        self._threshold_filter = SurprisalThresholdFilter(
            config=SurprisalThresholdConfig(enabled=False)
        )

        if requested_lexicon_mode:
            path = self._resolve_lexicon_path()
            if path is None:
                raise ValueError(
                    "Lexicon-backed candidates requested but no lexicon file was found. "
                    "Set LEXICON_PATH in config (or CLI) to an existing file, or place "
                    f"data/lexicon/lexicon_{self.cfg.language_code or '?'}.txt, "
                    "or disable lexicon mode (--no-use-lexicon-candidates / USE_LEXICON_CANDIDATES: false)."
                )
            self.lexicon_obj = Lexicon(str(path))
            self.use_lexicon = True
            self._lexicon_cache = LexiconCandidateCache(self.lexicon_obj)
            self.cfg.lexicon_path = str(path)
            print(f"[ChatAgent] Lexicon mode enabled: {path}")

        if not self.use_lexicon and self.generator_prompt is None:
            raise ValueError("Non-lexicon mode requires generator_prompt (DistractorGeneratorPrompt).")

        if self.cfg.apply_surprisal_threshold:
            try:
                scorer = SurprisalScorer(
                    model_id=resolve_surprisal_model_id(
                        primary_model_id=self.cfg.model_id,
                        selection_model_id=selection_mid,
                    ),
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
            except Exception as exc:
                print(
                    "[ChatAgent] WARNING: Failed to initialize surprisal scorer "
                    f"for model '{self.cfg.model_id}'. Surprisal filtering disabled. ({exc})"
                )

    # -------------------------
    # Helpers
    # -------------------------

    def _resolve_lexicon_path(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self.cfg.lexicon_path:
            candidates.append(Path(self.cfg.lexicon_path))
        if self.cfg.language_code:
            root = Path(__file__).resolve().parents[1]
            candidates.append(root / "data" / "lexicon" / f"lexicon_{self.cfg.language_code}.txt")
        for p in candidates:
            if p.exists():
                return p
        return None

    def _fallback_maze_out(self, core: str) -> Dict[str, Any]:
        return build_dummy_output(
            source=core,
            dummy_len=max(1, len(core) if core else 1),
            num_distractors=self.cfg.num_distractors,
        )

    def _reattach_punct(self, out: Dict[str, Any], prefix: str, suffix: str) -> Dict[str, Any]:
        return reattach_punctuation_to_output(out, prefix=prefix, suffix=suffix)

    def _openai_proposal_candidate_list(self, sentence_prefix: str, core: str) -> list[str]:
        """Merge up to :attr:`_GENERATOR_POOL_MAX_ROUNDS` OpenAI Stage-A generations for one (prefix, core)."""
        if self.generator_prompt is None:
            raise RuntimeError("generator_prompt is required in non-lexicon mode.")
        pad_token = "X" * max(1, len(core))
        raw_merged: list[str] = []
        cand_list: list[str] = []
        for _ in range(max(1, int(self._GENERATOR_POOL_MAX_ROUNDS))):
            raw_merged.extend(
                invoke_openai_generator(
                    render_messages=self.generator_prompt.render_messages,
                    chat_model=self.chat_gen,
                    sentence_prefix=sentence_prefix,
                    core=core,
                )
            )
            cand_list = apply_model_proposed_candidates(
                raw_merged,
                core,
                min_candidates=self.cfg.min_candidates,
                max_candidates=self.cfg.max_candidates,
            )
            n_non_pad = sum(
                1 for c in cand_list if str(c).strip() and str(c).strip() != pad_token
            )
            if n_non_pad >= int(self.cfg.min_candidates):
                break
        return cand_list

    def _resolve_candidates(
        self,
        target_word: str,
        *,
        sentence_prefix: Optional[str],
        defer_generator: bool,
    ) -> tuple[str, str, str, Sequence[str], Optional[int]]:
        """
        Shared path for lexicon vs model-proposed pools.

        When ``defer_generator`` is True (naturalistic runs without lexicon), returns an empty
        candidate tuple until Stage-A fills pools elsewhere.

        For generator-backed pools (naturalistic), the proposal model may be called up to
        :attr:`_GENERATOR_POOL_MAX_ROUNDS` times, merging raw word lists so new tokens can
        replace placeholder ``X`` padding until ``min_candidates`` non-placeholder entries exist
        (or rounds are exhausted).
        """
        pfx, core, sfx = strip_punctuation(target_word, self.puncts)
        if not core:
            return pfx, core, sfx, ("X" * len(target_word),), None
        forced_target = extract_forced_target(core)
        if forced_target is not None:
            return pfx, forced_target, sfx, tuple(), max(1, len(forced_target))

        if self.use_lexicon:
            if self._lexicon_cache is None:
                raise RuntimeError("Lexicon cache not initialized.")
            cand_list = lexicon_candidates(
                self._lexicon_cache,
                core,
                min_candidates=self.cfg.min_candidates,
                max_candidates=self.cfg.max_candidates,
            )
        else:
            if defer_generator:
                return pfx, core, sfx, tuple(), None
            if self.generator_prompt is None:
                raise RuntimeError("generator_prompt is required in non-lexicon mode.")
            if sentence_prefix is None:
                raise ValueError("sentence_prefix is required when building generator-based candidate pools.")
            cand_list = self._openai_proposal_candidate_list(sentence_prefix, core)

        cands = cand_list[: self.cfg.max_candidates]
        if len(cands) < self.cfg.min_candidates:
            pad = ["X" * len(core)] * max(0, self.cfg.min_candidates - len(cands))
            cands = (cands + pad)[: self.cfg.max_candidates]

        return pfx, core, sfx, cands, None

    def get_candidates(self, word: str) -> tuple[str, str, str, tuple[str, ...], Optional[int]]:
        pfx, core, sfx, cands, forced = self._resolve_candidates(
            word,
            sentence_prefix=None,
            defer_generator=not self.use_lexicon,
        )
        return pfx, core, sfx, tuple(cands), forced

    def _invoke_raw(self, sentence_prefix: str, target_word: str, candidates: Sequence[str]) -> str:
        sel_msgs = self.selector_prompt.render_messages(
            sentence_prefix=sentence_prefix, word=target_word, candidates=candidates
        )
        sel_resp = self.chat_sel.invoke(sel_msgs)
        return str(getattr(sel_resp, "content", sel_resp) or "")

    def _parse_to_dict(self, raw: str, *, repair: bool = True, core: Optional[str] = None) -> Dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            return self.selector_prompt.parser.parse(text).model_dump()
        except Exception:
            if not repair:
                return {}
        pad = "X" * max(1, len(core)) if core else "X"
        try:
            fixed = repair_json(text)
            obj = json.loads(fixed)
            if isinstance(obj, list):
                obj = next((x for x in reversed(obj) if isinstance(x, dict)), {})
            if not isinstance(obj, dict):
                return {}
            obj.setdefault("source", "")
            obj.setdefault("distractor1", pad)
            obj.setdefault("distractor2", pad)
            obj.setdefault("distractor3", None)
            return self.selector_prompt.parser.pydantic_object.model_validate(obj).model_dump()
        except Exception:
            return {}

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

    def read_controlled_input(self, path: Union[str, Path], split_on: Optional[str]) -> List[Dict[str, Any]]:
        return read_controlled_input_file(path, split_on)

    def run_controlled_experiment(
        self,
        controlled_items: Sequence[Dict[str, Any]],
        repair: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        validate_controlled_experiment_lexicon(self.use_lexicon)
        service = ControlledModeService(
            lexicon=self.lexicon_obj if self.use_lexicon else None,
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
            sorted_distractor_keys=sorted_distractor_keys,
        )
        return service.run(controlled_items=controlled_items, repair=repair, limit=limit)

    def _select_distractors(self, sentence_prefix: str, core: str, candidates: Sequence[str]) -> Dict[str, Any]:
        sel_msgs = self.selector_prompt.render_messages(sentence_prefix=sentence_prefix, word=core, candidates=candidates)
        sel_resp = self.chat_sel.invoke(sel_msgs)
        text = str(getattr(sel_resp, "content", sel_resp) or "")
        out = self._parse_to_dict(text, repair=True, core=core)
        if not out:
            return self._fallback_maze_out(core)
        return out

    # -------------------------
    # Jobs + Run
    # -------------------------

    def iter_jobs(self, input_data: Sequence[Sequence[str]]) -> Iterable[Dict[str, Any]]:
        for s_idx, tokens in enumerate(input_data):
            for i in range(len(tokens)):
                yield {
                    "sentence_index": s_idx,
                    "word_index": i,
                    "sentence_prefix": join_tokens(tokens[: i + 1], join_with=self.cfg.token_joiner, puncts=self.puncts),
                    "context_prefix": join_tokens(tokens[:i], join_with=self.cfg.token_joiner, puncts=self.puncts),
                    "target_word": tokens[i],
                }

    def run(self, input_data: Sequence[Sequence[str]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        run_start = time.perf_counter()
        total_sentences = len(input_data)
        results: List[Dict[str, Any]] = []
        current_sentence_index: Optional[int] = None
        current_words: List[Dict[str, Any]] = []

        for k, job in enumerate(self.iter_jobs(input_data)):
            if limit is not None and k >= limit:
                break

            if current_sentence_index is None:
                current_sentence_index = job["sentence_index"]
                print(f"[ChatAgent] Processing sentence: {current_sentence_index + 1}/{total_sentences}...")

            if job["sentence_index"] != current_sentence_index:
                results.append({"sentence_index": current_sentence_index, "words": current_words})
                current_sentence_index = job["sentence_index"]
                current_words = []
                print(f"[ChatAgent] Processing sentence: {current_sentence_index + 1}/{total_sentences}...")

            pfx, core, sfx, cands, forced_len = self._resolve_candidates(
                job["target_word"],
                sentence_prefix=job["sentence_prefix"],
                defer_generator=False,
            )
            if job["word_index"] == 0 or forced_len is not None:
                dummy_len = int(forced_len if forced_len is not None else len(job["target_word"]))
                out = build_dummy_output(
                    source=core,
                    dummy_len=max(1, dummy_len),
                    num_distractors=self.cfg.num_distractors,
                )
                out = ensure_distractor_count(
                    out,
                    num_distractors=self.cfg.num_distractors,
                    target_word=job["target_word"],
                    target_core=core,
                    candidate_pool=cands,
                    puncts=self.puncts,
                    allow_candidate_fill=False,
                )
                if forced_len is not None:
                    out = self._reattach_punct(out, pfx, sfx)
            else:
                out = self._select_distractors(job["sentence_prefix"], core, cands)
                out = ensure_distractor_count(
                    out,
                    num_distractors=self.cfg.num_distractors,
                    target_word=job["target_word"],
                    target_core=core,
                    candidate_pool=cands,
                    puncts=self.puncts,
                    allow_candidate_fill=True,
                )
                out = self._threshold_filter.enforce_output_threshold(
                    out=out,
                    context_prefix=job["context_prefix"],
                    target_core=core,
                    candidate_pool=cands,
                    puncts=self.puncts,
                    token_joiner=self.cfg.token_joiner,
                )
                out = self._reattach_punct(out, pfx, sfx)

            out.update(
                sentence_index=job["sentence_index"],
                word_index=job["word_index"],
                sentence_prefix=job["sentence_prefix"],
                target_word=job["target_word"],
            )
            current_words.append(out)

        if current_sentence_index is not None:
            results.append({"sentence_index": current_sentence_index, "words": current_words})

        print(f"[ChatAgent] Timing summary: total={time.perf_counter() - run_start:.2f}s")
        return results

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
# Minimal test for ChatAgent + save outputs
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat Maze distractor generation (generator + selector).")
    parser.add_argument("--config-path", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--language-code", default=None, help="Override LANGUAGE_CODE from config.")
    parser.add_argument(
        "--processing-mode",
        default=None,
        choices=["naturalistic_reading", "controlled_experiment"],
        help="Processing mode: naturalistic_reading or controlled_experiment (tabular item/condition/sentence input).",
    )
    parser.add_argument("--model-id", default=None, help="Override MODEL_ID from config.")
    parser.add_argument("--word-separator", default=None, help="Override WORD_SEPARATOR from config.")
    parser.add_argument("--output-path", default=None, help="Optional explicit output JSON path.")
    parser.add_argument("--input-data-path", default=None, help="Optional explicit input data path override.")
    parser.add_argument("--template-dir", default=None, help="Optional explicit template directory override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of word-jobs to run.")
    parser.add_argument("--num-distractors", type=int, default=None, help="Number of distractors per word (1-10).")
    parser.add_argument("--min-abs", type=float, default=None, help="Minimum absolute distractor surprisal.")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum surprisal delta over target.")
    parser.add_argument(
        "--absolute-threshold-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use only min_abs as hard threshold (ignore target + min_delta).",
    )
    parser.add_argument("--surprisal-device", default=None, help='Device for surprisal scorer, e.g. "cuda" or "cpu".')
    parser.add_argument("--lexicon-path", default=None, help="Optional explicit lexicon path override.")
    parser.add_argument(
        "--use-lexicon-candidates/--no-use-lexicon-candidates",
        dest="use_lexicon_candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Same flag family as api/llm_agent.py: use lexicon neighborhoods for Stage-A pools when "
            "the lexicon file exists; otherwise the generator proposes candidates. "
            "When omitted, follows USE_LEXICON_CANDIDATES in config.yaml (default True)."
        ),
    )
    parser.add_argument(
        "--proposal-model-id",
        default=None,
        help="Stage-A OpenAI model for candidate pools (defaults to MODEL_ID).",
    )
    parser.add_argument(
        "--selection-model-id",
        default=None,
        help="Stage-B OpenAI model for distractor selection (defaults to MODEL_ID).",
    )
    args = parser.parse_args()

    # Load API keys from common .env locations before initializing ChatOpenAI.
    try:
        from dotenv import load_dotenv

        load_dotenv()  # cwd + parent discovery
        project_env = Path(__file__).resolve().parents[1] / ".env"
        if project_env.exists():
            load_dotenv(project_env, override=False)
        config_env = Path(args.config_path).resolve().parent / ".env"
        if config_env.exists():
            load_dotenv(config_env, override=False)
    except Exception:
        # Keep behavior non-fatal if python-dotenv is unavailable.
        pass

    configs = load_config(args.config_path)
    if args.use_lexicon_candidates is not None:
        use_lexicon_candidates_cli = bool(args.use_lexicon_candidates)
    else:
        use_lexicon_candidates_cli = bool(configs.get("USE_LEXICON_CANDIDATES", True))
    language_code = str(args.language_code or configs.get("LANGUAGE_CODE", "en"))
    processing_mode = str(args.processing_mode or configs.get("PROCESSING_MODE", "naturalistic_reading")).strip().lower()
    if processing_mode not in ("naturalistic_reading", "controlled_experiment"):
        raise ValueError("processing_mode must be naturalistic_reading or controlled_experiment.")
    if processing_mode == "controlled_experiment":
        validate_controlled_experiment_lexicon(use_lexicon_candidates_cli)
    model_id = args.model_id or str(configs.get("MODEL_ID", "gpt-4o-mini"))
    mid_norm = str(model_id).strip()
    proposal_stage = configs.get("PROPOSAL_MODEL_ID")
    selection_stage = configs.get("SELECTION_MODEL_ID")
    if args.proposal_model_id is not None:
        proposal_stage = args.proposal_model_id
    if args.selection_model_id is not None:
        selection_stage = args.selection_model_id
    proposal_norm = str(proposal_stage or "").strip() or None
    selection_norm = str(selection_stage or "").strip() or None
    proposal_model_cfg = proposal_norm if proposal_norm and proposal_norm != mid_norm else None
    selection_model_cfg = selection_norm if selection_norm and selection_norm != mid_norm else None
    word_separator = args.word_separator if args.word_separator is not None else str(configs.get("WORD_SEPARATOR", " "))
    resolved_paths = resolve_runtime_paths(
        language_code=language_code,
        processing_mode=processing_mode,
        agent_type="chat",
        model_id=model_id,
        lexicon_path=args.lexicon_path or configs.get("LEXICON_PATH"),
        input_data_path=args.input_data_path or configs.get("INPUT_DATA_PATH"),
        template_dir=args.template_dir or configs.get("TEMPLATE_DIR"),
        output_path=args.output_path or configs.get("OUTPUT_PATH"),
    )
    if processing_mode == "controlled_experiment":
        input_data = None
    else:
        input_data = read_sentences_input(str(resolved_paths["input_data_path"]), split_on=word_separator)
    puncts = get_punctuation(language_code)
    token_join = token_joiner_for_language(language_code, word_separator=word_separator)
    num_distractors = int(args.num_distractors if args.num_distractors is not None else configs.get("NUM_DISTRACTORS", 3))
    min_abs = args.min_abs if args.min_abs is not None else configs.get("SURPRISAL_MIN_ABS")
    min_delta = float(args.min_delta if args.min_delta is not None else configs.get("SURPRISAL_MIN_DELTA", 0.0) or 0.0)
    if args.absolute_threshold_only is None:
        absolute_threshold_only = bool(configs.get("SURPRISAL_ABSOLUTE_THRESHOLD_ONLY", False))
    else:
        absolute_threshold_only = bool(args.absolute_threshold_only)
    surprisal_device = args.surprisal_device if args.surprisal_device is not None else configs.get("SURPRISAL_DEVICE")

    template_dir = Path(resolved_paths["template_dir"])
    if not template_dir.exists():
        raise ValueError(f"Template directory not found for LANGUAGE_CODE='{language_code}': {template_dir}")
    gen_prompt = DistractorGeneratorPrompt(
        path_to_user_template=template_dir / "distractor_gen_base.txt",
        path_to_extension_template=template_dir / "distractor_gen_extension.txt",
        path_to_system_template=template_dir / "system.txt",
    )
    selector_prompt = MazeChatPrompt(
        path_to_user_template=template_dir / "base.txt",
        path_to_extension_template=template_dir / "extension.txt",
        path_to_system_template=template_dir / "system.txt",
    )

    cfg = ChatAgentConfig(
        lexicon_path=resolved_paths["lexicon_path"],
        language_code=language_code,
        punctuations=puncts,
        num_distractors=num_distractors,
        token_joiner=token_join,
        lexicon_mode=use_lexicon_candidates_cli,
        apply_surprisal_threshold=(
            min_abs is not None
            or min_delta != 0.0
            or absolute_threshold_only
        ),
        min_abs=min_abs,
        min_delta=min_delta,
        absolute_threshold_only=absolute_threshold_only,
        surprisal_device=surprisal_device,
        model_id=model_id,
        proposal_model_id=proposal_model_cfg,
        selection_model_id=selection_model_cfg,
    )
    agent = ChatAgent(
        cfg=cfg,
        selector_prompt=selector_prompt,
        generator_prompt=gen_prompt,
    )

    if processing_mode == "controlled_experiment":
        controlled_items = agent.read_controlled_input(
            str(resolved_paths["input_data_path"]),
            split_on=word_separator,
        )
        outputs = agent.run_controlled_experiment(controlled_items, limit=args.limit)
    else:
        outputs = agent.run(input_data, limit=args.limit)
    output_path = args.output_path or resolved_paths["output_path"]
    agent.save_pretty_json(outputs, output_path)
    print(f"Saved output to: {output_path}")
