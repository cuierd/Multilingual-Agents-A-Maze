from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from json_repair import repair_json
from langchain_huggingface import HuggingFacePipeline

from utils.components import Lexicon, get_punctuation, strip_punctuation, attach_punctuation, normalize_candidates


@dataclass
class AgentConfig:
    model_id: str
    lexicon_path: str
    punctuations: Sequence[str]
    # generation params
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    return_full_text: bool = False
    # lexicon params
    min_candidates: int = 10
    max_candidates: int = 20


class LLMAgent:
    """
    For each sentence (list of tokens), iterate i=1..len-1:
      - prefix_sentence = ''.join(tokens[:i+1])  (ends at target word)
      - target_word = tokens[i]
      - candidates = lexicon neighbors for target_word (punct preserved)
      - LLM chooses distractors from candidates and returns schema-valid JSON
    """

    def __init__(self, prompt: Any, config: AgentConfig, llm: Optional[HuggingFacePipeline] = None) -> None:
        self.prompt = prompt
        self.cfg = config

        self.lexicon = Lexicon(self.cfg.lexicon_path)
        self.puncts = set(self.cfg.punctuations)
        self._neighbor_cache: dict[str, tuple[str, ...]] = {}

        self.llm = llm or HuggingFacePipeline.from_model_id(
            model_id=self.cfg.model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "do_sample": self.cfg.do_sample,
                "return_full_text": self.cfg.return_full_text,
            },
        )

    # ---------- helper function ----------
    def _get_cached_neighbors(self, core: str) -> tuple[str, ...]:
        if core not in self._neighbor_cache:
            raw = self.lexicon.get_neighbor(core, min_size=self.cfg.min_candidates, max_size=self.cfg.max_candidates)
            clean = normalize_candidates(raw)
            clean = [w for w in clean if w != core]
            self._neighbor_cache[core] = tuple(clean)
        return self._neighbor_cache[core]

    def _reattach_punct(self, out: Dict[str, Any], prefix: str, suffix: str) -> Dict[str, Any]:
        def wrap(x):
            if x is None:
                return None
            return attach_punctuation(x, prefix, suffix)
        for k in ("source", "distractor1", "distractor2", "distractor3"):
            if k in out:
                out[k] = wrap(out[k])
        return out

    # ---------- candidate generation ----------

    def get_candidates(self, word: str):
        prefix, core, suffix = strip_punctuation(word, self.puncts)
        if not core:
            return prefix, core, suffix, ("X" * len(word),)
        neighbors = self._get_cached_neighbors(core)
        return prefix, core, suffix, neighbors

    # ---------- LLM + parsing ----------

    def _invoke_raw(self, sentence_prefix: str, target_word: str, candidates: Sequence[str]) -> str:
        prompt_text = self.prompt.render_text(
            sentence_prefix=sentence_prefix,
            word=target_word,
            candidates=candidates,
        )
        resp = self.llm.invoke(prompt_text)
        return resp if isinstance(resp, str) else str(resp)

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
                sentence_prefix = "".join(tokens[: i + 1])
                pfx, core, sfx, candidates = self.get_candidates(target)
                yield {
                    "sentence_index": s_idx,
                    "word_index": i,
                    "sentence_prefix": sentence_prefix,
                    "target_word": target,
                    "target_core": core,
                    "punct_prefix": pfx,
                    "punct_suffix": sfx,
                    "candidates": candidates,
                }

    def run(
        self,
        input_data: Sequence[Sequence[str]],
        repair: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run over input_data and return a sentence-grouped list of JSON dict outputs."""
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
                out = {"source": job["target_core"], "distractor1": dummy, "distractor2": None, "distractor3": None}
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
                out = self._reattach_punct(out, job["punct_prefix"], job["punct_suffix"])
            # attach metadata (useful for reconstructing maze items)
            out.update(
                sentence_index=job["sentence_index"],
                word_index=job["word_index"],
                sentence_prefix=job["sentence_prefix"],
                target_word=job["target_word"],
            )
            current_words.append(out)
        if current_sentence_index is not None:
            results.append(
                {
                    "sentence_index": current_sentence_index,
                    "words": current_words,
                }
            )
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
# minimal usage example
# -------------------------
if __name__ == "__main__":
    from utils.components import load_config, read_sentences_input
    from utils.maze_prompt import MazeLLMPrompt  

    configs = load_config()
    language_code = configs["LANGUAGE_CODE"]
    agent_type = configs["AGENT_TYPE"]
    model_id = configs["MODEL_ID"]
    main_path = Path(f"/swdata/yin/Cui/LLM-MAZE/llmmaze/template/{language_code}")
    input_data = read_sentences_input(configs["INPUT_DATA_PATH"], split_on=configs["WORD_SEPARATOR"])
    puncts = get_punctuation(language_code)

    def _safe_name(s: str) -> str:
        # make filenames safe: "Qwen/Qwen2.5" -> "Qwen_Qwen2.5"
        return s.replace("/", "_").replace(" ", "_")
    model_tag = _safe_name(model_id)

    maze_prompt = MazeLLMPrompt(
        path_to_user_template=main_path / "base.txt",
        path_to_extension_template=main_path / "extension.txt",
        path_to_system_template=main_path / "system.txt",
    )

    agent_cfg = AgentConfig(
        model_id=model_id ,
        lexicon_path=configs["LEXICON_PATH"],
        punctuations=puncts,
    )

    agent = LLMAgent(prompt=maze_prompt, config=agent_cfg)

    # input_data = [
    #     ["人人", "对", "社会", "负有", "义务，", "因为", "只有", "在", "社会", "中", "他的", "个性", "才", "可能", "得到", "自由", "和", "充分的", "发展。"],
    #     ["学位", "流动性", "被", "定义", "为", "进入", "目的国", "的", "高等教育", "学位", "项目", "而", "跨越", "国境", "的", "实际", "行为。"],
    # ]

    # smoke test: run only first 5 jobs
    outputs = agent.run(input_data, limit=None)
    print(outputs)
    agent.save_pretty_json(outputs, f"./results/zh/{agent_type}_out_{language_code}_{model_tag}.json")
