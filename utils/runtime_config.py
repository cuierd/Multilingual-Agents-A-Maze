from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


_DEFAULTS: Dict[str, Any] = {
    "PROCESSING_MODE": "naturalistic_reading",
    "LANGUAGE_CODE": "en",
    "AGENT_TYPE": "llm",
    "MODEL_ID": "Qwen/Qwen2.5-3B-Instruct",
    # If empty, fall back to MODEL_ID (same model proposes and selects).
    "PROPOSAL_MODEL_ID": "",
    "SELECTION_MODEL_ID": "",
    # False = Stage-A generates candidates (no lexicon neighborhoods).
    "USE_LEXICON_CANDIDATES": True,
    # If empty, paths are auto-derived from LANGUAGE_CODE and PROCESSING_MODE.
    "LEXICON_PATH": "",
    "INPUT_DATA_PATH": "",
    "TEMPLATE_DIR": "",
    "OUTPUT_PATH": "",
    "WORD_SEPARATOR": " ",
    "NUM_DISTRACTORS": 3,
    "NUM_WORKERS": 1,
    # Surprisal threshold controls (optional)
    "SURPRISAL_MIN_ABS": None,
    "SURPRISAL_MIN_DELTA": 0.0,
    "SURPRISAL_ABSOLUTE_THRESHOLD_ONLY": False,
    "SURPRISAL_DEVICE": None,
}


def validate_controlled_experiment_lexicon(use_lexicon_candidates: bool) -> None:
    """
    Controlled experiments rely on lexicon-backed neighbor pooling across conditions.
    Fail fast when callers disable lexicon candidates.
    """
    if bool(use_lexicon_candidates):
        return
    raise ValueError(
        "Controlled experiment mode requires lexicon-backed candidates. "
        "Set USE_LEXICON_CANDIDATES=true or pass --use-lexicon-candidates; "
        "do not use --no-use-lexicon-candidates."
    )


def _merge_layers(defaults: Dict[str, Any], yaml_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config layers with precedence: defaults < yaml_cfg < overrides."""
    merged = dict(defaults)
    merged.update({k: v for k, v in yaml_cfg.items() if v is not None})
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


@dataclass(frozen=True)
class LLMRuntimeConfig:
    processing_mode: str
    language_code: str
    agent_type: str
    model_id: str
    proposal_model_id: str
    selection_model_id: str
    use_lexicon_candidates: bool
    lexicon_path: str
    input_data_path: str
    template_dir: str
    output_path: str
    word_separator: str
    num_distractors: int
    num_workers: int
    min_abs: Optional[float]
    min_delta: float
    absolute_threshold_only: bool
    surprisal_device: Optional[str]

    @property
    def apply_surprisal_threshold(self) -> bool:
        if self.absolute_threshold_only:
            return True
        if self.min_abs is not None:
            return True
        return float(self.min_delta) != 0.0
    # convert the dataclass to a dictionary
    def to_public_dict(self) -> Dict[str, Any]:
        return asdict(self)   


def _safe_model_tag(model_id: str) -> str:
    return str(model_id).replace("/", "_").replace(" ", "_")


def _as_abs_path(path_text: Optional[str]) -> Optional[Path]:
    text = str(path_text or "").strip()
    if not text:
        return None
    p = Path(text).expanduser()
    if p.is_absolute():
        return p
    # cwd > project_root
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / p).resolve()


def resolve_runtime_paths(
    *,
    language_code: str,
    processing_mode: str,
    agent_type: str,
    model_id: str,
    lexicon_path: Optional[str] = None,
    input_data_path: Optional[str] = None,
    template_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, str]:
    code = str(language_code).strip().lower()
    mode = str(processing_mode).strip().lower()
    agent = str(agent_type).strip().lower()

    preferred_input_name = (
        f"controlled_{code}.txt"
        if mode == "controlled_experiment"
        else f"naturalistic_{code}.txt"
    )
    preferred_input_path = PROJECT_ROOT / "data" / "input" / code / preferred_input_name
    default_input_path = preferred_input_path
    default_lexicon_path = PROJECT_ROOT / "data" / "lexicon" / f"lexicon_{code}.txt"
    default_template_dir = PROJECT_ROOT / "template" / code
    default_output_path = PROJECT_ROOT / "results" / code / f"{agent}_out_{code}_{_safe_model_tag(model_id)}.json"

    resolved_lexicon = _as_abs_path(lexicon_path) or default_lexicon_path
    resolved_input = _as_abs_path(input_data_path) or default_input_path
    resolved_template = _as_abs_path(template_dir) or default_template_dir
    resolved_output = _as_abs_path(output_path) or default_output_path

    return {
        "lexicon_path": str(resolved_lexicon),
        "input_data_path": str(resolved_input),
        "template_dir": str(resolved_template),
        "output_path": str(resolved_output),
    }


def build_llm_runtime_config(
    yaml_cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> LLMRuntimeConfig:
    merged = _merge_layers(_DEFAULTS, yaml_cfg or {}, overrides or {})
    processing_mode = str(merged.get("PROCESSING_MODE", "naturalistic_reading")).strip().lower()
    if processing_mode not in {"naturalistic_reading", "controlled_experiment"}:
        raise ValueError("PROCESSING_MODE must be 'naturalistic_reading' or 'controlled_experiment'.")
    use_lexicon_candidates = bool(merged.get("USE_LEXICON_CANDIDATES", True))
    if processing_mode == "controlled_experiment":
        validate_controlled_experiment_lexicon(use_lexicon_candidates)
    language_code = str(merged.get("LANGUAGE_CODE", "en")).strip()
    agent_type = str(merged.get("AGENT_TYPE", "llm")).strip()
    model_id = str(merged.get("MODEL_ID", _DEFAULTS["MODEL_ID"])).strip()
    proposal_model_id = str(merged.get("PROPOSAL_MODEL_ID", "") or "").strip() or model_id
    selection_model_id = str(merged.get("SELECTION_MODEL_ID", "") or "").strip() or model_id
    paths = resolve_runtime_paths(
        language_code=language_code,
        processing_mode=processing_mode,
        agent_type=agent_type,
        model_id=model_id,
        lexicon_path=merged.get("LEXICON_PATH"),
        input_data_path=merged.get("INPUT_DATA_PATH"),
        template_dir=merged.get("TEMPLATE_DIR"),
        output_path=merged.get("OUTPUT_PATH"),
    )

    min_abs = merged.get("SURPRISAL_MIN_ABS")
    min_abs = None if min_abs is None else float(min_abs)

    min_delta = float(merged.get("SURPRISAL_MIN_DELTA", 0.0) or 0.0)
    absolute_only = bool(merged.get("SURPRISAL_ABSOLUTE_THRESHOLD_ONLY", False))
    num_distractors = int(merged.get("NUM_DISTRACTORS", 3))
    if not 1 <= num_distractors <= 10:
        raise ValueError("NUM_DISTRACTORS must be an integer in [1, 10].")
    num_workers = int(merged.get("NUM_WORKERS", 1))
    if num_workers < 1:
        raise ValueError("NUM_WORKERS must be an integer >= 1.")

    surprisal_device = str(merged.get("SURPRISAL_DEVICE") or "").strip() or None

    return LLMRuntimeConfig(
        processing_mode=processing_mode,
        language_code=language_code,
        agent_type=agent_type,
        model_id=model_id,
        proposal_model_id=proposal_model_id,
        selection_model_id=selection_model_id,
        use_lexicon_candidates=use_lexicon_candidates,
        lexicon_path=paths["lexicon_path"],
        input_data_path=paths["input_data_path"],
        template_dir=paths["template_dir"],
        output_path=paths["output_path"],
        word_separator=str(merged.get("WORD_SEPARATOR", " ")),
        num_distractors=num_distractors,
        num_workers=num_workers,
        min_abs=min_abs,
        min_delta=min_delta,
        absolute_threshold_only=absolute_only,
        surprisal_device=surprisal_device,
    )
