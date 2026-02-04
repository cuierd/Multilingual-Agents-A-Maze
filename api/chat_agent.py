from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from json_repair import repair_json
from langchain_openai import ChatOpenAI

from utils.components import (
    Lexicon,
    strip_punctuation,
    attach_punctuation,
    normalize_candidates,
)


@dataclass
class ChatAgentConfig:
    lexicon_path: Optional[str] = None
    punctuations: Sequence[str] = ()
    min_candidates: int = 10
    max_candidates: int = 20
    model_id: str = "gpt-4o-mini"
    gen_temperature: float = 0.7
    sel_temperature: float = 0.2


class ChatAgent:
    def __init__(
        self,
        cfg: ChatAgentConfig,
        selector_prompt: Any,
        generator_prompt: Optional[Any] = None,
        *,
        lexicon: Optional[bool] = None,
        chat_gen: Optional[ChatOpenAI] = None,
        chat_sel: Optional[ChatOpenAI] = None,
    ) -> None:
        self.cfg = cfg
        self.selector_prompt = selector_prompt
        self.generator_prompt = generator_prompt
        self.use_lexicon = bool(lexicon) if lexicon is not None else False
        self.puncts = set(self.cfg.punctuations)
        self.chat_gen = chat_gen or ChatOpenAI(model=self.cfg.model_id, temperature=self.cfg.gen_temperature)
        self.chat_sel = chat_sel or ChatOpenAI(model=self.cfg.model_id, temperature=self.cfg.sel_temperature)

        self.lexicon_obj: Optional[Lexicon] = None
        self._neighbor_cache: dict[str, tuple[str, ...]] = {}

        if self.use_lexicon:
            if not self.cfg.lexicon_path:
                raise ValueError("lexicon=True requires cfg.lexicon_path to be set.")
            self.lexicon_obj = Lexicon(self.cfg.lexicon_path)

        if not self.use_lexicon and self.generator_prompt is None:
            raise ValueError("lexicon is None/False requires generator_prompt (DistractorGeneratorPrompt).")

    # -------------------------
    # Helpers
    # -------------------------

    def _fallback_maze_out(self, core: str) -> Dict[str, Any]:
        dummy = "X" * (len(core) if core else 1)
        return {"source": core, "distractor1": dummy, "distractor2": dummy, "distractor3": None}

    def _reattach_punct(self, out: Dict[str, Any], prefix: str, suffix: str) -> Dict[str, Any]:
        def wrap(x):
            if x is None:
                return None
            return attach_punctuation(x, prefix, suffix)

        for k in ("source", "distractor1", "distractor2", "distractor3"):
            if k in out:
                out[k] = wrap(out[k])
        return out

    def _get_cached_neighbors(self, core: str) -> tuple[str, ...]:
        if not self.lexicon_obj:
            raise RuntimeError("Lexicon object not initialized (lexicon mode is off).")

        if core not in self._neighbor_cache:
            raw = self.lexicon_obj.get_neighbor(core, min_size=self.cfg.min_candidates, max_size=self.cfg.max_candidates)
            clean = normalize_candidates(raw)
            clean = [w for w in clean if w and w != core][: self.cfg.max_candidates]
            self._neighbor_cache[core] = tuple(clean)
        return self._neighbor_cache[core]

    def _parse_distractor_pool_lenient(self, text: str) -> List[str]:
        """Return list[str] or [] if anything goes wrong."""
        try:
            fixed = repair_json(text)
            obj = json.loads(fixed)
            words = obj.get("distractors") if isinstance(obj, dict) else obj if isinstance(obj, list) else []
            return [w for w in words if isinstance(w, str)]
        except Exception:
            return []

    def _get_candidates(self, target_word: str, sentence_prefix: str) -> tuple[str, str, str, Sequence[str]]:
        pfx, core, sfx = strip_punctuation(target_word, self.puncts)
        if not core:
            return pfx, core, sfx, ("X" * len(target_word),)

        if self.use_lexicon:
            cands = list(self._get_cached_neighbors(core))
        else:
            gen_msgs = self.generator_prompt.render_messages(sentence_prefix=sentence_prefix, word=core)
            gen_resp = self.chat_gen.invoke(gen_msgs)
            words = self._parse_distractor_pool_lenient(gen_resp.content)
            cands = normalize_candidates(words)
            cands = [c for c in cands if c != core]

        # enforce size + fallback if too few
        cands = cands[: self.cfg.max_candidates]
        if len(cands) < self.cfg.min_candidates:
            # keep pipeline running (fallback placeholders)
            pad = ["X" * len(core)] * max(0, self.cfg.min_candidates - len(cands))
            cands = (cands + pad)[: self.cfg.max_candidates]

        return pfx, core, sfx, cands

    def _select_distractors(self, sentence_prefix: str, core: str, candidates: Sequence[str]) -> Dict[str, Any]:
        sel_msgs = self.selector_prompt.render_messages(sentence_prefix=sentence_prefix, word=core, candidates=candidates)
        sel_resp = self.chat_sel.invoke(sel_msgs)
        text = sel_resp.content

        # 1) strict parse
        try:
            out = self.selector_prompt.parser.parse(text).model_dump()
            return out
        except Exception:
            pass

        # 2) repair + lenient dict + fill keys
        try:
            fixed = repair_json(text)
            obj = json.loads(fixed)
            if isinstance(obj, list):
                obj = next((x for x in reversed(obj) if isinstance(x, dict)), {})
            if not isinstance(obj, dict):
                return self._fallback_maze_out(core)

            obj.setdefault("source", core)
            obj.setdefault("distractor1", "X" * (len(core) if core else 1))
            obj.setdefault("distractor2", "X" * (len(core) if core else 1))
            obj.setdefault("distractor3", None)

            out = self.selector_prompt.parser.pydantic_object.model_validate(obj).model_dump()
            return out
        except Exception:
            return self._fallback_maze_out(core)

    # -------------------------
    # Jobs + Run
    # -------------------------

    def iter_jobs(self, input_data: Sequence[Sequence[str]]) -> Iterable[Dict[str, Any]]:
        for s_idx, tokens in enumerate(input_data):
            for i in range(len(tokens)):
                yield {
                    "sentence_index": s_idx,
                    "word_index": i,
                    "sentence_prefix": "".join(tokens[: i + 1]),
                    "target_word": tokens[i],
                }

    def run(self, input_data: Sequence[Sequence[str]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
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

            if job["word_index"] == 0:
                dummy = "X" * len(job["target_word"])
                out = {"source": job["target_word"], "distractor1": dummy, "distractor2": None, "distractor3": None}
            else:
                pfx, core, sfx, cands = self._get_candidates(job["target_word"], job["sentence_prefix"])
                out = self._select_distractors(job["sentence_prefix"], core, cands)
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
    import json
    from pathlib import Path
    from dotenv import load_dotenv

    from utils.components import load_config, get_punctuation, read_sentences_input
    from utils.maze_prompt import DistractorGeneratorPrompt, MazeChatPrompt
    from chat_agent import ChatAgent, ChatAgentConfig

    load_dotenv()
    configs = load_config()

    language_code = configs["LANGUAGE_CODE"]
    agent_type = "OpenAI"
    model_id = "gpt-4o-mini"
    template_dir = Path(f"/swdata/yin/Cui/LLM-MAZE/llmmaze/template/{language_code}")
    puncts = get_punctuation(language_code)


    print("Language:", language_code)
    print("Template dir:", template_dir)

    def _safe_name(s: str) -> str:
        # make filenames safe: "Qwen/Qwen2.5" -> "Qwen_Qwen2.5"
        return s.replace("/", "_").replace(" ", "_")

    # -------------------------------------------------
    # Input (tokenized sentences)
    # -------------------------------------------------

    input_data = read_sentences_input(configs["INPUT_DATA_PATH"], split_on=configs["WORD_SEPARATOR"])

    # -------------------------------------------------
    # Build Prompt Objects
    # -------------------------------------------------
    gen_prompt = DistractorGeneratorPrompt(
        path_to_user_template=template_dir / "chat_distractor_gen_base.txt",
        path_to_extension_template=template_dir / "chat_distractor_gen_extension.txt",
        path_to_system_template=template_dir / "system.txt",
    )

    selector_prompt = MazeChatPrompt(
        path_to_user_template=template_dir / "base.txt",
        path_to_extension_template=template_dir / "extension.txt",
        path_to_system_template=template_dir / "system.txt",
    )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------
    out_dir = Path(f"./results/{language_code}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = _safe_name(model_id)

    # =================================================
    # TEST 1: Pure LLM mode (Generator + Selector)
    # =================================================
    print("\n" + "=" * 80)
    print("TEST 1: ChatAgent (LLM → LLM)")
    print("=" * 80)

    cfg_llm = ChatAgentConfig(
        lexicon_path=None,
        punctuations=puncts,
        model_id="gpt-4o-mini",
        gen_temperature=0.7,
        sel_temperature=0.2,
    )

    agent_llm = ChatAgent(
        cfg=cfg_llm,
        selector_prompt=selector_prompt,
        generator_prompt=gen_prompt,
        lexicon=None,  # important: LLM→LLM mode
    )

    out_llm = agent_llm.run(input_data, limit=None)

    llm_json = out_dir / f"chat_nonlex_out_{language_code}_{model_tag}.json"
    agent_llm.save_pretty_json(out_llm, llm_json)
    print("Saved JSON :", llm_json)

    # =================================================
    # TEST 2: Lexicon + LLM selector
    # =================================================
    print("\n" + "=" * 80)
    print("TEST 2: ChatAgent (Lexicon → LLM)")
    print("=" * 80)

    cfg_lex = ChatAgentConfig(
        lexicon_path=configs["LEXICON_PATH"],
        punctuations=puncts,
        model_id="gpt-4o-mini",
        sel_temperature=0.2,
    )

    agent_lex = ChatAgent(
        cfg=cfg_lex,
        selector_prompt=selector_prompt,
        generator_prompt=None,  # not used
        lexicon=True,           # enable lexicon mode
    )

    out_lex = agent_lex.run(input_data, limit=None)

    print("\n--- Lexicon Mode Output ---")
    lex_json = out_dir / f"chat_lex_out_{language_code}_{model_tag}.json"
    agent_lex.save_pretty_json(out_lex, lex_json)

    print("Saved JSON :", lex_json)

    print("\n✅ ChatAgent test finished.")
