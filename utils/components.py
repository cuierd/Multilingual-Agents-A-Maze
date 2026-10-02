from __future__ import annotations
import pandas as pd
import random
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable, Sequence
from functools import lru_cache
import yaml
import string
_RNG = random.Random(42)
# random.seed(42)

_BASE_PUNCT = {
    "latin": set(string.punctuation) | {"“", "”", "‘", "’", "—", "–", "…", "«", "»", "„"},
    "zh": {"。", "，", "！", "？", "：", "；", "、", "（", "）", "《", "》",
           "「", "」", "『", "』", "【", "】", "“", "”", "‘", "’", "—", "…"},
    "ja": {"。", "、", "！", "？", "「", "」", "『", "』", "・", "ー", "…"},
    "ko": set(string.punctuation) | {"…", "·", "“", "”"},
    "arabic": set(string.punctuation) | {
        "،", "؛", "؟", "٪", "٫", "٬",
        "«", "»", "“", "”", "‘", "’",
        "…", "—", "–",
    },
}

_LANG_MAP = {
    "en": "latin", "de": "latin", "fr": "latin", "es": "latin", "hr": "latin", "ru": "latin",
    "zh": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-yue": "zh", "zh-tw": "zh", "zh-hk": "zh", "zh-yu": "zh", "yue": "zh", "yu": "zh",
    "ja": "ja", "ko": "ko",
    "ar": "arabic", "fa": "arabic", "ur": "arabic",
}


def get_punctuation(
    language_code: str = "latin",
    extra: Optional[Iterable[str]] = None,
    remove: Optional[Iterable[str]] = None,
) -> Set[str]:
    code = str(language_code or "latin").strip().lower()
    key = _LANG_MAP.get(code, code)
    punct = set(_BASE_PUNCT.get(key, _BASE_PUNCT["latin"]))

    if extra:
        punct |= set(extra)
    if remove:
        punct -= set(remove)

    return punct


@lru_cache(maxsize=None)
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}

_NO_SPACE_LANGS = {"chinese", "zh", "japanese", "ja", "thai", "th", "cantonese", "yue", "yu"}


def normalize_wordfreq_language_code(language_code: str) -> str:
    """Map wordfreq/HF-style tags onto a coarse code for typography rules (e.g. zh-cn -> zh)."""
    s = str(language_code or "").strip().lower()
    if not s:
        return ""
    if s.startswith("zh"):
        return "zh"
    return s.split("-", 1)[0]


def language_uses_concatenated_surface_form(language_code: str) -> bool:
    """
    True for orthographies whose standard typography omits ASCII spaces between word-like units.
    Inputs are still split on WORD_SEPARATOR / whitespace in pipelines; joiners revert to contiguous
    text for these languages (e.g. Chinese, Japanese, Thai).
    """
    return normalize_wordfreq_language_code(language_code) in _NO_SPACE_LANGS


def token_joiner_for_language(language_code: str, *, word_separator: str = " ") -> str:
    """
    String used between surface tokens when reconstructing prefixes for LM/surprisal.
    """
    # return "" if language_uses_concatenated_surface_form(language_code) else str(word_separator)
    return "" if language_uses_concatenated_surface_form(language_code) else " "


def _split_sentence(sentence, split_on=None):
    if sentence is None:
        raise ValueError("There is no sentence to split.")
    if split_on is None:
        return list(sentence)
    if split_on == "":
        raise ValueError("split_on cannot be an empty string.")
    return sentence.split(split_on)

def sentences_to_word_lists(sentences, split_on=None):
    if not sentences:
        return []
    return [_split_sentence(sentence, split_on) for sentence in sentences]

_NO_SPACE_BEFORE = {
    ".", ",", "!", "?", ":", ";",
    "。", "，", "！", "？", "：", "；", "、",
    "،", "؛", "؟", "٪", "٫", "٬",
    ")", "]", "}", "）", "】", "》", "」", "』",
    "»", "”", "’",
}
_NO_SPACE_AFTER = {
    "(", "[", "{", "（", "【", "《", "「", "『",
    "«", "“", "‘",  "„",
}

def join_tokens(tokens: Sequence[str], join_with: str = " ", puncts: Optional[Set[str]] = None) -> str:
    """
    Join tokens into a sentence with optional punctuation-aware spacing.
    If join_with == " " and puncts is provided, suppress space before common closing punctuation.
    """
    if not tokens:
        return ""
    if join_with == "":
        return "".join(tokens)
    if join_with != " " or puncts is None:
        return join_with.join(tokens)

    out = str(tokens[0])
    prev = str(tokens[0])
    for tok in tokens[1:]:
        tok = str(tok)
        is_punct_token = bool(tok) and all(ch in puncts for ch in tok)
        if is_punct_token and all(ch in _NO_SPACE_BEFORE for ch in tok):
            out += tok
        elif prev and all(ch in _NO_SPACE_AFTER for ch in prev):
            out += tok
        else:
            out += " " + tok
        prev = tok
    return out

def _read_lines_from_txt(path_to_data):
    with open(path_to_data, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

# TODO: check if this function is needed.
def _read_rows_from_csv(path_to_data):
    with open(path_to_data, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return [row for row in reader if row]

def read_sentences_input(data_input, split_on=None):
    if isinstance(data_input, os.PathLike):
        data_input = os.fspath(data_input)

    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise ValueError(f"Input file does not exist: {data_input}")
        _, ext = os.path.splitext(data_input)
        if ext.lower() == ".txt":
            return sentences_to_word_lists(_read_lines_from_txt(data_input), split_on=split_on)
        if ext.lower() == ".csv":
            return _read_rows_from_csv(data_input)
        raise ValueError(f"Unsupported file type: {ext.lower()}")

    if isinstance(data_input, list):
        if not data_input:
            return []
        if all(isinstance(x, list) for x in data_input):
            return data_input
        if all(isinstance(x, str) for x in data_input):
            return sentences_to_word_lists(data_input, split_on=split_on)
        raise ValueError("List input must be list[str] or list[list[str]].")

    raise ValueError("data_input must be a file path or a list input.")

def strip_punctuation(word: str, puncts: Set[str]) -> Tuple[str, str, str]:
    start = 0
    end = len(word)
    # leading
    while start < end and word[start] in puncts:
        start += 1
    # trailing
    while end > start and word[end - 1] in puncts:
        end -= 1
    return word[:start], word[start:end], word[end:]

def attach_punctuation(core: str, prefix: str, suffix: str) -> str:
    return f"{prefix}{core}{suffix}"

def normalize_candidates(words: Sequence[str]) -> list[str]:
    """Strip, drop empties, de-duplicate (order-preserving), trim."""
    uniq = []
    seen = set()
    for w in words:
        if not w:
            continue
        w = w.strip()
        if not w:
            continue
        if w in seen:
            continue
        seen.add(w)
        uniq.append(w)
    return uniq


@dataclass
class Lexicon:
    path_to_lexicon: str
    rank_bin: int = 100

    def __post_init__(self) -> None:
        if self.rank_bin < 1:
            raise ValueError("rank_bin must be an integer >= 1.")
        df = pd.read_csv(
            self.path_to_lexicon,
            sep=self._infer_sep(self.path_to_lexicon),
            engine="python",
            encoding="utf-8",
        )
        df.columns = [c.lower().strip() for c in df.columns]
        if "word" not in df.columns or "frequency_rank" not in df.columns:
            raise ValueError("Lexicon must contain columns: 'word' and 'frequency_rank'.")

        # Normalize words early so len() is always safe and malformed rows are removed.
        df["word"] = df["word"].fillna("").astype(str).str.strip()
        df = df[df["word"] != ""]
        # Guard against common stringified-null artifacts from CSV/Arrow parsing.
        df = df[~df["word"].str.lower().isin({"nan", "none", "<na>"})]
        df["frequency_rank"] = pd.to_numeric(df["frequency_rank"], errors="coerce")
        df = df.dropna(subset=["frequency_rank"])
        df["frequency_rank"] = df["frequency_rank"].astype(int)

        if "length" not in df.columns:
            df["length"] = df["word"].map(len)
        else:
            fallback_length = df["word"].map(lambda x: len(x) if isinstance(x, str) else 0)
            df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(fallback_length).astype(int)

        df["freq_group"] = ((df["frequency_rank"] - 1) // self.rank_bin).astype(int)

        self.df = df
        if df.empty:
            raise ValueError("Lexicon contains no valid rows after cleaning.")
        self.max_freq_group = int(df["freq_group"].max())
        self.max_frequency_rank = int(df["frequency_rank"].max())
        self.min_length = int(df["length"].min())
        self.max_length = int(df["length"].max())
        self.group_to_words: Dict[Tuple[int, int], Set[str]] = (
            df.groupby(["length", "freq_group"])["word"].apply(set).to_dict()
        )
        self.word_to_group: Dict[str, Tuple[int, int]] = dict(
            zip(df["word"], zip(df["length"], df["freq_group"]))
        )
        # keep best (smallest) rank for each word
        rank_series = df.groupby("word")["frequency_rank"].min()
        self.word_to_rank: Dict[str, int] = {w: int(r) for w, r in rank_series.items()}

    @staticmethod
    def _infer_sep(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
        if "\t" in header:
            return "\t"
        if "," in header:
            return ","
        return r"\s+"

    def get_neighbor(self, word: str, min_size: int = 10, max_size: Optional[int] = None) -> List[str]:
        word = word.strip()
        if not word:
            return []

        w_len, fg = self.word_to_group.get(word, (len(word), self.max_freq_group))
        out: Set[str] = set()
        delta = 0

        while len(out) < min_size:
            left, right = fg - delta, fg + delta
            if left < 0 and right > self.max_freq_group:
                break
            if 0 <= left <= self.max_freq_group:
                out |= self.group_to_words.get((w_len, left), set())
            if right != left and 0 <= right <= self.max_freq_group:
                out |= self.group_to_words.get((w_len, right), set())
            out.discard(word)
            delta += 1

        neighbors = list(out)
        if max_size is not None and len(neighbors) > max_size:
            # random.shuffle(neighbors)
            _RNG.shuffle(neighbors)
            neighbors = neighbors[:max_size]
        return neighbors

    def get_rank(self, word: str, default_to_max: bool = True) -> Optional[int]:
        rank = self.word_to_rank.get(word)
        if rank is not None:
            return int(rank)
        if default_to_max:
            return int(self.max_frequency_rank)
        return None

    def get_neighbor_by_profile(
        self,
        *,
        target_length: int,
        target_rank: int,
        min_size: int = 10,
        max_size: Optional[int] = None,
        exclude_words: Optional[Iterable[str]] = None,
        max_length_delta: int = 0,
    ) -> List[str]:
        """
        Retrieve neighbors by target length/rank profile (without a pivot word).
        Useful for controlled items where target words differ across conditions.
        """
        out: Set[str] = set()
        excludes = set(exclude_words or [])
        t_len = max(self.min_length, min(self.max_length, int(target_length)))
        rank = max(1, min(self.max_frequency_rank, int(target_rank)))
        fg = (rank - 1) // self.rank_bin

        for freq_delta in range(0, self.max_freq_group + 1):
            for len_delta in range(0, max_length_delta + 1):
                lengths = [t_len] if len_delta == 0 else [t_len - len_delta, t_len + len_delta]
                groups = [fg] if freq_delta == 0 else [fg - freq_delta, fg + freq_delta]
                for l in lengths:
                    if l < self.min_length or l > self.max_length:
                        continue
                    for g in groups:
                        if g < 0 or g > self.max_freq_group:
                            continue
                        out |= self.group_to_words.get((l, g), set())
            out -= excludes
            if len(out) >= min_size:
                break

        neighbors = list(out)
        if max_size is not None and len(neighbors) > max_size:
            random.shuffle(neighbors)
            neighbors = neighbors[:max_size]
        return neighbors
            
        
if __name__ == "__main__":
    lexicon = Lexicon("/swdata/yin/Cui/LLM-MAZE/llmmaze/data/lexicon/lexicon_zh.txt")
    neighbors = lexicon.get_neighbor("人权",max_size=20)
    print(neighbors)

    # puncts_zh = get_punct("zh")

    # puncts_en = get_punct("en", extra={"★"}, remove={"'"})

    # print("Punctuations for Chinese:")
    # print(puncts_zh)
    # print("Punctuations for English:")
    # print(puncts_en)

    # word = "人权,。？"
    # pure, punct, suffix = strip_punctuation(word, puncts_zh)
    # print(pure, "->", punct, "->", suffix)

    # word = "‘’rights,."
    # pure, punct, suffix = strip_punctuation(word, puncts_en)
    # print(pure, "->", punct, "->", suffix)

    # word = "rig.hts,★"
    # pure, punct, suffix = strip_punctuation(word, puncts_en)
    # print(pure, "->", punct, "->", suffix)
