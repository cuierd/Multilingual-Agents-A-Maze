from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

import gradio as gr


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from api.chat_agent import ChatAgent, ChatAgentConfig
from api.llm_agent import AgentConfig, LLMAgent
from utils.components import get_punctuation, load_config, read_sentences_input, token_joiner_for_language
from utils.distractor_utils import sorted_distractor_keys
from utils.maze_prompt import DistractorGeneratorPrompt, MazeChatPrompt, MazeLLMPrompt
from utils.runtime_config import validate_controlled_experiment_lexicon

_CACHE_LOCK = threading.RLock()
_LLM_AGENT_CACHE: dict[str, Any] = {"key": None, "agent": None}
_CHAT_AGENT_CACHE: dict[str, Any] = {"key": None, "agent": None}


def _parse_optional_float(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _resolve_lexicon_path(language_code: str, cfg: dict, uploaded_lexicon: Optional[str]) -> Optional[str]:
    if uploaded_lexicon:
        p = Path(uploaded_lexicon)
        return str(p) if p.exists() else None

    cfg_path = str(cfg.get("LEXICON_PATH", "")).strip()
    if cfg_path:
        p = Path(cfg_path)
        if p.exists():
            return str(p)

    default_path = PROJECT_DIR / "data" / "lexicon" / f"lexicon_{language_code}.txt"
    if default_path.exists():
        return str(default_path)
    return None


def _load_input_text(text_input: str, file_input: Optional[str]) -> tuple[str, Optional[str]]:
    if file_input:
        p = Path(file_input)
        if not p.exists():
            raise ValueError(f"Uploaded file does not exist: {file_input}")
        return p.read_text(encoding="utf-8"), str(p)

    text = str(text_input or "").strip()
    if not text:
        raise ValueError("Please type sentences or upload a file.")
    return text, None


def _build_template_dir(language_code: str) -> Path:
    p = PROJECT_DIR / "template" / language_code
    if not p.exists():
        raise ValueError(f"Template folder not found for language code '{language_code}': {p}")
    return p


def _save_temp_json(data: list[dict]) -> str:
    fd, path = tempfile.mkstemp(prefix="llmmaze_out_", suffix=".json")
    os.close(fd)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _save_temp_text(content: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix="llmmaze_out_", suffix=suffix)
    os.close(fd)
    Path(path).write_text(content, encoding="utf-8")
    return path


def _auto_surprisal_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _condition_words_cell(word_data: dict[str, Any]) -> str:
    condition_words = word_data.get("condition_words")
    if not isinstance(condition_words, dict) or not condition_words:
        return ""
    seen: list[str] = []
    for value in condition_words.values():
        token = str(value)
        if token not in seen:
            seen.append(token)
    if not seen:
        return ""
    if len(seen) == 1:
        return seen[0]
    return "|".join(seen)


def _outputs_to_tsv(outputs: list[dict]) -> str:
    rows: list[dict[str, str]] = []
    max_distractors = 0
    for record in outputs:
        for word_data in record.get("words", []):
            target_cell = _condition_words_cell(word_data)
            if not target_cell:
                target_cell = str(word_data.get("target_word", word_data.get("source", "")))
            distractor_keys = sorted_distractor_keys(word_data)
            max_distractors = max(max_distractors, len(distractor_keys))
            row: dict[str, str] = {"target": target_cell}
            for idx, key in enumerate(distractor_keys, start=1):
                row[f"distractor{idx}"] = str(word_data.get(key, ""))
            rows.append(row)

    output = io.StringIO()
    columns = ["target"] + [f"distractor{i}" for i in range(1, max_distractors + 1)]
    writer = csv.DictWriter(output, fieldnames=columns, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _discover_language_codes() -> list[str]:
    template_dir = PROJECT_DIR / "template"
    if not template_dir.exists():
        return []
    return sorted([p.name for p in template_dir.iterdir() if p.is_dir()])


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _explicit_hf_stage_or_none(override_text: str, *, primary_model: str) -> Optional[str]:
    text = str(override_text or "").strip()
    primary = str(primary_model or "").strip()
    return text if text and text != primary else None


def _build_llm_agent(
    *,
    model_id: str,
    proposal_model_id: str,
    selection_model_id: str,
    use_lexicon_candidates: bool,
    lex_path: Optional[str],
    puncts: set[str],
    num_distractors: int,
    num_workers: int,
    apply_surprisal: bool,
    min_abs: Optional[float],
    min_delta: float,
    absolute_threshold_only: bool,
    surprisal_device: str,
    token_joiner: str,
    template_dir: Path,
) -> LLMAgent:
    maze_prompt = MazeLLMPrompt(
        path_to_user_template=template_dir / "base.txt",
        path_to_extension_template=template_dir / "extension.txt",
        path_to_system_template=template_dir / "system.txt",
    )
    generator_prompt = None
    if not bool(use_lexicon_candidates):
        generator_prompt = DistractorGeneratorPrompt(
            path_to_user_template=template_dir / "distractor_gen_base.txt",
            path_to_extension_template=template_dir / "distractor_gen_extension.txt",
            path_to_system_template=template_dir / "system.txt",
        )
    primary = str(model_id).strip()
    agent_cfg = AgentConfig(
        model_id=primary,
        proposal_model_id=_explicit_hf_stage_or_none(proposal_model_id, primary_model=primary),
        selection_model_id=_explicit_hf_stage_or_none(selection_model_id, primary_model=primary),
        lexicon_path=(
            (str(lex_path).strip() if lex_path else None) if bool(use_lexicon_candidates) else None
        ),
        punctuations=puncts,
        num_distractors=int(num_distractors),
        num_workers=int(max(1, num_workers)),
        apply_surprisal_threshold=apply_surprisal,
        min_abs=min_abs,
        min_delta=min_delta,
        absolute_threshold_only=bool(absolute_threshold_only),
        surprisal_device=surprisal_device,
        token_joiner=token_joiner,
        use_lexicon_candidates=bool(use_lexicon_candidates),
        generator_prompt=generator_prompt,
    )
    return LLMAgent(prompt=maze_prompt, cfg=agent_cfg)


def _build_chat_agent(
    *,
    model_id: str,
    proposal_model_id: str,
    selection_model_id: str,
    language_code: str,
    puncts: set[str],
    num_distractors: int,
    token_joiner: str,
    lexicon_mode: bool,
    lex_path: Optional[str],
    apply_surprisal: bool,
    min_abs: Optional[float],
    min_delta: float,
    absolute_threshold_only: bool,
    surprisal_device: str,
    template_dir: Path,
) -> ChatAgent:
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
    primary = str(model_id).strip()
    chat_cfg = ChatAgentConfig(
        lexicon_path=lex_path,
        language_code=language_code,
        punctuations=puncts,
        min_candidates=10,
        max_candidates=20,
        num_distractors=int(num_distractors),
        token_joiner=token_joiner,
        lexicon_mode=bool(lexicon_mode),
        apply_surprisal_threshold=apply_surprisal,
        min_abs=min_abs,
        min_delta=min_delta,
        absolute_threshold_only=bool(absolute_threshold_only),
        surprisal_device=surprisal_device,
        model_id=primary,
        proposal_model_id=_explicit_hf_stage_or_none(proposal_model_id, primary_model=primary),
        selection_model_id=_explicit_hf_stage_or_none(selection_model_id, primary_model=primary),
    )
    return ChatAgent(
        cfg=chat_cfg,
        selector_prompt=selector_prompt,
        generator_prompt=gen_prompt,
        lexicon_mode=bool(lexicon_mode),
    )


def run_web_generation(
    config_path: str,
    agent_type: str,
    processing_mode: str,
    language_code: str,
    model_id: str,
    num_distractors: int,
    lexicon_mode: bool,
    lexicon_file: Optional[str],
    hf_use_lexicon_candidates: bool,
    proposal_model_override: str,
    selection_model_override: str,
    min_abs_text: str,
    min_delta: float,
    absolute_threshold_only: bool,
    output_format: str,
    text_input: str,
    file_input: Optional[str],
):
    start = time.perf_counter()
    try:
        cfg = load_config(config_path)

        language_code = str(language_code or cfg.get("LANGUAGE_CODE", "zh")).strip()
        model_id = str(model_id or cfg.get("MODEL_ID", "")).strip()
        pm_eff = str(proposal_model_override or "").strip() or str(cfg.get("PROPOSAL_MODEL_ID", "") or "").strip()
        sm_eff = str(selection_model_override or "").strip() or str(cfg.get("SELECTION_MODEL_ID", "") or "").strip()
        # Web app input convention: words must be whitespace-tokenized.
        word_separator = " "
        agent_type = str(agent_type or "LLM").strip().lower()
        processing_mode = str(processing_mode or "NATURALISTIC_READING").strip().lower()
        output_format = str(output_format or "TSV").strip().upper()
        min_abs = _parse_optional_float(min_abs_text)
        min_delta = float(min_delta or 0.0)
        cfg_sd = str(cfg.get("SURPRISAL_DEVICE") or "").strip()
        surprisal_device = cfg_sd if cfg_sd else _auto_surprisal_device()
        apply_surprisal = (min_abs is not None) or (min_delta != 0.0) or bool(absolute_threshold_only)

        # Match CLI: controlled experiments always require lexicon-backed candidate pools.
        if processing_mode == "controlled_experiment":
            if agent_type == "llm":
                validate_controlled_experiment_lexicon(bool(hf_use_lexicon_candidates))
            else:
                validate_controlled_experiment_lexicon(bool(lexicon_mode))
        token_joiner = token_joiner_for_language(language_code, word_separator=word_separator)

        puncts = get_punctuation(language_code)
        template_dir = _build_template_dir(language_code)
        raw_text, uploaded_path = _load_input_text(text_input, file_input)
        input_path = uploaded_path

        if processing_mode == "controlled_experiment":
            if input_path is None:
                # controlled parser expects file-style tabular rows
                fd, tmp_path = tempfile.mkstemp(prefix="llmmaze_controlled_", suffix=".txt")
                os.close(fd)
                Path(tmp_path).write_text(raw_text, encoding="utf-8")
                input_path = tmp_path
            input_data = None
        else:
            # For naturalistic mode, parse line-wise to support both typed text and uploaded files.
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            input_data = read_sentences_input(lines, split_on=word_separator)

        cache_hit = False
        if agent_type == "llm":
            use_hf_lex = bool(hf_use_lexicon_candidates)
            lex_path_llm = _resolve_lexicon_path(language_code, cfg, lexicon_file) if use_hf_lex else None
            if use_hf_lex and not lex_path_llm:
                raise ValueError(
                    "Could not find lexicon file for HF (lexicon mode). Upload a lexicon or uncheck "
                    "'Use lexicon neighborhoods for HF candidate pools'."
                )
            proposal_key = pm_eff if pm_eff != model_id else ""
            selection_key = sm_eff if sm_eff != model_id else ""
            llm_key = (
                model_id,
                proposal_key,
                selection_key,
                bool(use_hf_lex),
                str(lex_path_llm or ""),
                int(num_distractors),
                int(max(1, cfg.get("NUM_WORKERS", 1))),
                bool(apply_surprisal),
                min_abs,
                float(min_delta),
                bool(absolute_threshold_only),
                str(surprisal_device),
                token_joiner,
                str(template_dir),
                tuple(sorted(puncts)),
            )
            with _CACHE_LOCK:
                if _LLM_AGENT_CACHE["key"] == llm_key and _LLM_AGENT_CACHE["agent"] is not None:
                    agent = _LLM_AGENT_CACHE["agent"]
                    cache_hit = True
                else:
                    old_agent = _LLM_AGENT_CACHE.get("agent")
                    _LLM_AGENT_CACHE["agent"] = None
                    _LLM_AGENT_CACHE["key"] = None
                    if old_agent is not None:
                        del old_agent
                        _release_model_memory()
                    agent = _build_llm_agent(
                        model_id=model_id,
                        proposal_model_id=pm_eff,
                        selection_model_id=sm_eff,
                        use_lexicon_candidates=use_hf_lex,
                        lex_path=lex_path_llm,
                        puncts=puncts,
                        num_distractors=int(num_distractors),
                        num_workers=int(max(1, cfg.get("NUM_WORKERS", 1))),
                        apply_surprisal=apply_surprisal,
                        min_abs=min_abs,
                        min_delta=float(min_delta),
                        absolute_threshold_only=bool(absolute_threshold_only),
                        surprisal_device=surprisal_device,
                        token_joiner=token_joiner,
                        template_dir=template_dir,
                    )
                    _LLM_AGENT_CACHE["agent"] = agent
                    _LLM_AGENT_CACHE["key"] = llm_key
            if processing_mode == "controlled_experiment":
                if not input_path:
                    raise ValueError("Controlled mode requires tabular input (upload file or tabular text).")
                controlled_items = agent.read_controlled_input(input_path, split_on=word_separator)
                outputs = agent.run_controlled_experiment(controlled_items)
            else:
                outputs = agent.run(input_data)
        else:
            lex_path = _resolve_lexicon_path(language_code, cfg, lexicon_file) if lexicon_mode else None
            if lexicon_mode and not lex_path:
                raise ValueError(
                    "Could not find lexicon file for CHAT (lexicon mode enabled). Upload a lexicon or turn off "
                    "lexicon-backed candidate pools."
                )
            proposal_key_chat = pm_eff if pm_eff != model_id else ""
            selection_key_chat = sm_eff if sm_eff != model_id else ""
            chat_key = (
                model_id,
                proposal_key_chat,
                selection_key_chat,
                language_code,
                bool(lexicon_mode),
                str(lex_path or ""),
                int(num_distractors),
                bool(apply_surprisal),
                min_abs,
                float(min_delta),
                bool(absolute_threshold_only),
                str(surprisal_device),
                token_joiner,
                str(template_dir),
                tuple(sorted(puncts)),
            )
            with _CACHE_LOCK:
                if _CHAT_AGENT_CACHE["key"] == chat_key and _CHAT_AGENT_CACHE["agent"] is not None:
                    agent = _CHAT_AGENT_CACHE["agent"]
                    cache_hit = True
                else:
                    old_agent = _CHAT_AGENT_CACHE.get("agent")
                    _CHAT_AGENT_CACHE["agent"] = None
                    _CHAT_AGENT_CACHE["key"] = None
                    if old_agent is not None:
                        del old_agent
                        _release_model_memory()
                    agent = _build_chat_agent(
                        model_id=model_id,
                        proposal_model_id=pm_eff,
                        selection_model_id=sm_eff,
                        language_code=language_code,
                        puncts=puncts,
                        num_distractors=int(num_distractors),
                        token_joiner=token_joiner,
                        lexicon_mode=bool(lexicon_mode),
                        lex_path=lex_path,
                        apply_surprisal=apply_surprisal,
                        min_abs=min_abs,
                        min_delta=float(min_delta),
                        absolute_threshold_only=bool(absolute_threshold_only),
                        surprisal_device=surprisal_device,
                        template_dir=template_dir,
                    )
                    _CHAT_AGENT_CACHE["agent"] = agent
                    _CHAT_AGENT_CACHE["key"] = chat_key
            if processing_mode == "controlled_experiment":
                if not input_path:
                    raise ValueError("Controlled mode requires tabular input (upload file or tabular text).")
                controlled_items = agent.read_controlled_input(input_path, split_on=word_separator)
                outputs = agent.run_controlled_experiment(controlled_items)
            else:
                outputs = agent.run(input_data)

        if output_format == "TSV":
            out_text = _outputs_to_tsv(outputs)
            out_file = _save_temp_text(out_text, ".tsv")
        else:
            out_text = json.dumps(outputs, ensure_ascii=False, indent=2)
            out_file = _save_temp_json(outputs)
        elapsed = time.perf_counter() - start
        status = (
            f"Done in {elapsed:.2f}s | agent={agent_type.upper()} | mode={processing_mode.upper()} | "
            f"records={len(outputs)} | cache={'HIT' if cache_hit else 'MISS'}"
        )
        return status, out_text, out_file
    except Exception as exc:
        elapsed = time.perf_counter() - start
        status = f"Error after {elapsed:.2f}s: {exc.__class__.__name__}: {exc}"
        err_json = json.dumps(
            {
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
            ensure_ascii=False,
            indent=2,
        )
        return status, err_json, None


def build_app() -> gr.Blocks:
    default_config = str(PROJECT_DIR / "config.yaml")
    language_choices = _discover_language_codes() or ["zh"]
    try:
        _boot_cfg = load_config(default_config)
        lexicon_ui_default = bool(_boot_cfg.get("USE_LEXICON_CANDIDATES", True))
    except Exception:
        lexicon_ui_default = True
    css = """
    #app-title { text-align: center; margin-bottom: 0.2rem; }
    #app-subtitle { text-align: center; margin-top: 0; color: #666; }
    #download-wrap { display: flex; justify-content: center; }
    #download-btn { min-width: 280px; }
    """
    with gr.Blocks(title="Agentic-Maze: new generation of Maze task distractor generation toolkit", css=css) as app:
        gr.HTML("<h1 id='app-title'>Agentic-Maze: new generation of Maze task distractor generation toolkit</h1>")
        gr.HTML(
            "<p id='app-subtitle'>Multilingual distractor generation for naturalistic reading and controlled experiments with Maze task.</p>"
        )
        gr.Markdown(
            "- **AGENT TYPE**: **LLM** (RECOMMENDED, local Hugging Face model + lexicon) or **CHAT** (OpenAI API).\n"
            "- **LANGUAGE CODE**: Use [wordfreq language codes](https://pypi.org/project/wordfreq/) such as `en`, `zh`, `fa`.\n"
            "- **MODEL_ID**: For LLM use a Hugging Face model ID (e.g. `Qwen/Qwen2.5-3B-Instruct`); for CHAT use an OpenAI model (e.g. `gpt-4o-mini`).\n"
            "- **Controlled experiments** require lexicon-backed candidates: HF checkbox **HF_USE_LEXICON_CANDIDATES** ON, Chat **USE LEXICON MATCHING** ON, with a usable lexicon file."
        )
        config_path = gr.State(default_config)
        with gr.Row():
            agent_type = gr.Dropdown(
                label="AGENT TYPE",
                choices=["LLM", "CHAT"],
                value="LLM",
                info="LLM is recommended for stable lexicon-based multilingual runs.",
            )
            processing_mode = gr.Dropdown(
                label="EXPERIMENT MODE",
                choices=["naturalistic_reading", "controlled_experiment"],
                value="naturalistic_reading",
                info="naturalistic_reading: plain sentence list. controlled_experiment: tabular item/condition/sentence input.",
            )
        with gr.Row():
            language_code = gr.Dropdown(
                label="LANGUAGE CODE",
                choices=language_choices,
                value="zh" if "zh" in language_choices else language_choices[0],
                info="Use codes consistent with wordfreq (e.g., en, zh, fa).",
            )
            model_id = gr.Textbox(
                label="MODEL_ID",
                value="Qwen/Qwen2.5-3B-Instruct",
                info="Find LLM model IDs on Hugging Face model pages; CHAT model IDs from OpenAI docs.",
            )
        proposal_model_tb = gr.Textbox(
            label="PROPOSAL_MODEL_ID (optional Stage-A)",
            value="",
            info=(
                "Optional. Leave blank to use MODEL_ID (Recommended). "
                "Used only in Stage A when lexicon-backed candidates are OFF. Please provide a valid Hugging Face model ID if AGENT TYPE is LLM."
                "Use OpenAI model ID if AGENT TYPE is CHAT."
            ),
        )
        selection_model_tb = gr.Textbox(
            label="SELECTION_MODEL_ID (optional Stage-B)",
            value="",
            info=(
                "Optional. Leave blank to use MODEL_ID (Recommended). "
                "Used only in Stage B. Please provide a valid Hugging Face model ID if AGENT TYPE is LLM."
                "Use OpenAI model ID if AGENT TYPE is CHAT."
            ),
        )
        with gr.Row():
            num_distractors = gr.Slider(label="Num Distractors", minimum=1, maximum=10, step=1, value=3)
            hf_lexicon_candidates = gr.Checkbox(
                label="LLM USE LEXICON CANDIDATES",
                value=lexicon_ui_default,
                info="LLM agent: recommended ON. "
                "Controlled mode requires this ON.",
            )
            lexicon_mode = gr.Checkbox(
                label="CHAT USE LEXICON MATCHING",
                value=lexicon_ui_default,
                info="Chat agent: recommended ON. "
                "Controlled mode requires this ON.",
            )
            output_format = gr.Dropdown(
                label="OUTPUT FORMAT",
                choices=["JSON", "TSV"],
                value="TSV",
                info="TSV format: first column target/condition words, following columns distractors.",
            )
        with gr.Row():
            min_abs_text = gr.Textbox(
                label="MIN_ABS",
                value="",
                info="Minimum absolute distractor surprisal. Leave blank to disable.",
            )
            min_delta = gr.Number(
                label="MIN_DELTA",
                value=0.0,
                info="Minimum surprisal gap: distractor surp. - target surp. Leave 0 to disable.",
            )
            absolute_threshold_only = gr.Checkbox(
                label="ABSOLUTE THRESHOLD ONLY",
                value=False,
                info="If ON, only MIN_ABS is enforced; MIN_DELTA is ignored.",
            )
        lexicon_file = gr.File(
            label="Optional Lexicon Upload (.txt/.tsv/.csv)",
            type="filepath",
            file_types=[".txt", ".tsv", ".csv"],
        )
        gr.Markdown(
            "If uploaded, this lexicon file is used first; otherwise the app uses config/default lexicon path."
        )
        gr.Markdown("### Input")
        gr.Markdown(
            "**Input requirement:** words must be separated by whitespace, regardless of language. Please mark numbers, symbols, etc. with `**`, e.g., `**2026**`, so that the app will not generate distractors for them."
        )
   
   
        text_input = gr.Textbox(
            label="Sentence(s) or tabular controlled text",
            lines=8,
            placeholder="Naturalistic: one tokenized sentence per line.\nControlled: tabular rows (item_id/condition_id/sentence).",
        )
        file_input = gr.File(label="Or upload input file (.txt/.tsv/.csv)", type="filepath")
        run_btn = gr.Button("Run Generation", variant="primary")
        gr.Markdown("### Output")
        status = gr.Textbox(label="Status")
        output_json = gr.Textbox(label="Output Preview", lines=20)
        with gr.Row(elem_id="download-wrap"):
            output_file = gr.DownloadButton(
                label="Download Output File",
                value=None,
                elem_id="download-btn",
                variant="primary",
            )

        run_btn.click(
            fn=run_web_generation,
            inputs=[
                config_path,
                agent_type,
                processing_mode,
                language_code,
                model_id,
                num_distractors,
                lexicon_mode,
                lexicon_file,
                hf_lexicon_candidates,
                proposal_model_tb,
                selection_model_tb,
                min_abs_text,
                min_delta,
                absolute_threshold_only,
                output_format,
                text_input,
                file_input,
            ],
            outputs=[status, output_json, output_file],
        )
    return app


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the LLM Maze Gradio web app.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

