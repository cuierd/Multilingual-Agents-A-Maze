import pandas as pd
import random
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Iterable, Sequence
from functools import lru_cache
import yaml
import string
random.seed(42)

_BASE_PUNCT = {
    "latin": set(string.punctuation) | {"“", "”", "‘", "’", "—", "–", "…"},
    "zh": {"。", "，", "！", "？", "：", "；", "、", "（", "）", "《", "》",
           "「", "」", "『", "』", "【", "】", "“", "”", "‘", "’", "—", "…"},
    "ja": {"。", "、", "！", "？", "「", "」", "『", "』", "・", "ー", "…"},
    "ko": set(string.punctuation) | {"…", "·", "“", "”"},
}

_LANG_MAP = {
    "en": "latin", "de": "latin", "fr": "latin", "es": "latin",
    "zh": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh",
    "ja": "ja", "ko": "ko",
}


def get_punctuation(
    language_code: str = "latin",
    extra: Optional[Iterable[str]] = None,
    remove: Optional[Iterable[str]] = None,
) -> Set[str]:

    key = _LANG_MAP.get(language_code.lower(), language_code.lower())
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

def load_punctuation(punctuation_list):
    if not punctuation_list:
        return frozenset()
    return frozenset(punctuation_list)

_NO_SPACE_LANGS = {"chinese", "zh", "japanese", "ja", "thai", "th"}

def _default_separator(language, fallback=" "):
    if not language:
        return fallback
    normalized = str(language).strip().lower()
    return "" if normalized in _NO_SPACE_LANGS else fallback

def _split_sentence(sentence, split_on=None):
    if sentence is None:
        raise ValueError("There is no sentence to split.")
    if split_on is None:
        return list(sentence)
    if split_on:
        return sentence.split(split_on)

def sentences_to_word_lists(sentences, split_on=None):
    if not sentences:
        return []
    return [_split_sentence(sentence, split_on) for sentence in sentences]

def _combine_words(words, join_with=" ", language=None):
    if words is None:
        raise ValueError("There is no words to combine.")
    if language is not None:
        join_with = _default_separator(language, fallback=join_with)
    return join_with.join(words)

def word_lists_to_sentences(word_lists: list[list[str]], join_with=" ", language=None):
    if not word_lists:
        raise ValueError("There is no word lists to combine.")
    return [_combine_words(words, join_with, language=language) for words in word_lists]

def _read_lines_from_txt(path_to_data):
    with open(path_to_data, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

def _read_rows_from_csv(path_to_data):
    with open(path_to_data, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return [row for row in reader if row]

def read_sentences_input(data_input, split_on=None):
    if isinstance(data_input, str) and os.path.exists(data_input):
        _, ext = os.path.splitext(data_input)
        if ext.lower() == ".txt":
            return sentences_to_word_lists(_read_lines_from_txt(data_input), split_on=split_on)
        if ext.lower() == ".csv":
            return sentences_to_word_lists(_read_rows_from_csv(data_input), split_on=split_on)
        raise ValueError(f"Unsupported file type: {ext.lower()}")

    if isinstance(data_input, list):
        # list of word lists
        if data_input and all(isinstance(x, list) for x in data_input):
            return data_input
        # list of sentences (strings)
        if data_input and all(isinstance(x, str) for x in data_input):
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
        df = pd.read_csv(
            self.path_to_lexicon,
            sep=self._infer_sep(self.path_to_lexicon),
            engine="python",
            encoding="utf-8",
        )
        df.columns = [c.lower().strip() for c in df.columns]
        if "word" not in df.columns or "frequency_rank" not in df.columns:
            raise ValueError("Lexicon must contain columns: 'word' and 'frequency_rank'.")

        df["word"] = df["word"].astype(str)
        df["frequency_rank"] = pd.to_numeric(df["frequency_rank"], errors="coerce")
        df = df.dropna(subset=["frequency_rank"])
        df["frequency_rank"] = df["frequency_rank"].astype(int)

        if "length" not in df.columns:
            df["length"] = df["word"].map(len)
        else:
            df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(df["word"].map(len)).astype(int)

        df["freq_group"] = ((df["frequency_rank"] - 1) // self.rank_bin).astype(int)

        self.df = df
        self.max_freq_group = int(df["freq_group"].max())
        self.group_to_words: Dict[Tuple[int, int], Set[str]] = (
            df.groupby(["length", "freq_group"])["word"].apply(set).to_dict()
        )
        self.word_to_group: Dict[str, Tuple[int, int]] = dict(
            zip(df["word"], zip(df["length"], df["freq_group"]))
        )

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