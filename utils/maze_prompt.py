from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, conlist


# ======================================================
# Output Schemas
# ======================================================

class MazeOutput(BaseModel):
    source: str = Field(description="the target source word")
    distractor1: str = Field(description="the first distractor word")
    distractor2: str = Field(description="the second distractor word")
    distractor3: Optional[str] = Field(description="the third distractor word (optional)")


class DistractorPoolOutput(BaseModel):
    distractors: conlist(str, min_length=10, max_length=20) = Field(
        ..., description="List of 10–20 unique distractor words."
    )


# ======================================================
# Base Prompt Class (Shared Logic)
# ======================================================

class BaseChatPrompt:
    """
    Shared infrastructure for all maze-related chat prompts.
    """

    def __init__(
        self,
        path_to_user_template: Union[str, Path],
        path_to_extension_template: Optional[Union[str, Path]] = None,
        path_to_system_template: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
    ) -> None:

        self.user = PromptTemplate.from_template(
            Path(path_to_user_template).read_text(encoding=encoding)
        )

        self.extension = (
            PromptTemplate.from_template(
                Path(path_to_extension_template).read_text(encoding=encoding)
            ).partial(CANDIDATES="")
            if path_to_extension_template
            else None
        )

        self.system_text = (
            Path(path_to_system_template).read_text(encoding=encoding)
            if path_to_system_template
            else None
        )

    @staticmethod
    def _format_candidates(candidates: Optional[Union[str, Sequence[str]]]) -> str:
        """Format candidates as one-per-line text."""
        if not candidates:
            return ""

        if isinstance(candidates, str):
            return candidates

        return "\n".join(f"- {w}" for w in candidates)

    def _build_chat(
        self,
        human_template: str,
    ) -> ChatPromptTemplate:

        msgs = []

        if self.system_text:
            msgs.append(("system", self.system_text))

        msgs.append(("human", human_template))

        return ChatPromptTemplate.from_messages(msgs)


# ======================================================
# Classic Maze Prompt (Single-Step)
# ======================================================

class MazeLLMPrompt(BaseChatPrompt):
    """
    One-shot Maze prompt (legacy / compatibility).
    """

    def __init__(
        self,
        path_to_user_template: Union[str, Path],
        path_to_extension_template: Optional[Union[str, Path]],
        path_to_system_template: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
    ) -> None:

        super().__init__(
            path_to_user_template,
            path_to_extension_template,
            path_to_system_template,
            encoding,
        )

        self.parser = PydanticOutputParser(pydantic_object=MazeOutput)

        human_template = (
            "## Task\n{USER_PROMPT}\n\n"
            "## Instructions\n{EXTENSION}\n\n"
            "## Output format\n{FORMAT_INSTRUCTIONS}"
        )

        self.chat = self._build_chat(human_template)

    def render_messages(
        self,
        sentence_prefix: str,
        word: str,
        candidates: Optional[Union[str, Sequence[str]]] = None,
    ) -> List:

        user_prompt = self.user.format(
            SENTENCE_PREFIX=sentence_prefix,
            WORD=word,
            EXTENSION="",
        )

        candidates_block = self._format_candidates(candidates)

        extension_prompt = (
            self.extension.format(
                WORD=word,
                CANDIDATES=candidates_block,
            )
            if self.extension
            else ""
        )

        return self.chat.format_messages(
            USER_PROMPT=user_prompt,
            EXTENSION=extension_prompt,
            FORMAT_INSTRUCTIONS=self.parser.get_format_instructions(),
        )

    def render_text(
        self,
        sentence_prefix: str,
        word: str,
        candidates: Optional[Union[str, Sequence[str]]] = None,
    ) -> str:
        """Flatten chat messages into a single string prompt (for HF text-generation)."""
        msgs = self.render_messages(sentence_prefix, word, candidates)
        system = ""
        user = ""
        for m in msgs:
            if m.type == "system":
                system = m.content.strip()
            elif m.type == "human":
                user = m.content.strip()
        if system:
            return system + "\n\n" + user
        return user


# ======================================================
# Stage A: Generator
# ======================================================

class DistractorGeneratorPrompt(BaseChatPrompt):
    """
    Stage A: Generate 10–20 distractor candidates.
    """

    def __init__(
        self,
        path_to_user_template: Union[str, Path],
        path_to_extension_template: Union[str, Path],
        path_to_system_template: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
    ) -> None:

        super().__init__(
            path_to_user_template,
            path_to_extension_template,
            path_to_system_template,
            encoding,
        )

        self.max_size = 20
        self.parser = PydanticOutputParser(pydantic_object=DistractorPoolOutput)

        human_template = (
            "## Task\n{USER_PROMPT}\n\n"
            "## Constraints\n{EXTENSION}\n\n"
            "## Output format\n{FORMAT_INSTRUCTIONS}"
        )

        self.chat = self._build_chat(human_template)

    def render_messages(
        self,
        sentence_prefix: str,
        word: str,
    ) -> List:

        user_prompt = self.user.format(
            SENTENCE_PREFIX=sentence_prefix,
            WORD=word,
            EXTENSION="",
        )

        extension_prompt = self.extension.format(
            WORD=word,
            MAX_SIZE=self.max_size,
        )

        return self.chat.format_messages(
            USER_PROMPT=user_prompt,
            EXTENSION=extension_prompt,
            FORMAT_INSTRUCTIONS=self.parser.get_format_instructions(),
        )

    def render_text(
        self,
        sentence_prefix: str,
        word: str,
    ) -> str:
        """Flatten chat messages into a single string prompt (for HF text-generation)."""
        msgs = self.render_messages(sentence_prefix, word)
        system = ""
        user = ""
        for m in msgs:
            if m.type == "system":
                system = m.content.strip()
            elif m.type == "human":
                user = m.content.strip()
        if system:
            return system + "\n\n" + user
        return user


# ======================================================
# Stage B: Maze Chat Prompt (Stage A + Stage B)
# ======================================================

class MazeChatPrompt(BaseChatPrompt):
    """
    Stage B: Select 3 worst continuations.
    """

    def __init__(
        self,
        path_to_user_template: Union[str, Path],
        path_to_extension_template: Union[str, Path],
        path_to_system_template: Optional[Union[str, Path]] = None,
        encoding: str = "utf-8",
    ) -> None:

        super().__init__(
            path_to_user_template,
            path_to_extension_template,
            path_to_system_template,
            encoding,
        )

        self.parser = PydanticOutputParser(pydantic_object=MazeOutput)

        human_template = (
            "## Task\n{USER_PROMPT}\n\n"
            "## Instructions\n{EXTENSION}\n\n"
            "## Output format\n{FORMAT_INSTRUCTIONS}"
        )

        self.chat = self._build_chat(human_template)

    def render_messages(
        self,
        sentence_prefix: str,
        word: str,
        candidates: Union[str, Sequence[str]],
    ) -> List:

        user_prompt = self.user.format(
            SENTENCE_PREFIX=sentence_prefix,
            WORD=word,
            EXTENSION="",
        )

        candidates_block = self._format_candidates(candidates)

        extension_prompt = self.extension.format(
            WORD=word,
            SENTENCE_PREFIX=sentence_prefix,
            CANDIDATES=candidates_block,
        )

        return self.chat.format_messages(
            USER_PROMPT=user_prompt,
            EXTENSION=extension_prompt,
            FORMAT_INSTRUCTIONS=self.parser.get_format_instructions(),
        )



def main():
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    from utils.components import load_config, normalize_candidates

    load_dotenv()

    print("CWD:", os.getcwd())
    configs = load_config()
    print("Configs:", configs)

    language_code = configs["LANGUAGE_CODE"]
    template_dir = Path(f"/home/cding/projects/Agentic-Maze/template/{language_code}")
    print("Template path:", template_dir)

    # -------------------------
    # Example input
    # -------------------------
    sentence_prefix = "人人对社会负有义务"
    word = "义务"

    # -------------------------
    # LLM: use same model twice (gen + select)
    # -------------------------
    gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    sel_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # -------------------------
    # Prompt objects
    # -------------------------
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

    oneshot_prompt = MazeLLMPrompt(
        path_to_user_template=template_dir / "base.txt",
        path_to_extension_template=template_dir / "extension.txt",  
        path_to_system_template=template_dir / "system.txt",
    )

    # =========================================================
    # 1) Stage A: generator
    # =========================================================
    print("\n" + "=" * 80)
    print("[TEST 1] DistractorGeneratorPrompt (Stage A)")
    print("=" * 80)

    gen_msgs = gen_prompt.render_messages(sentence_prefix, word)
    gen_resp = gen_llm.invoke(gen_msgs)

    print("\n--- Raw Stage A output ---\n")
    print(gen_resp.content)

    pool = gen_prompt.parser.parse(gen_resp.content)
    candidates = normalize_candidates(pool.distractors)

    # remove target if it accidentally appears
    candidates = [c for c in candidates if c != word]

    print(f"\n--- Parsed candidates (n={len(candidates)}) ---")
    for i, c in enumerate(candidates, 1):
        print(f"{i:02d}. {c}")

    if len(candidates) < 10:
        raise RuntimeError(f"Stage A returned too few candidates: {len(candidates)}")

    # =========================================================
    # 2) Stage B: selector
    # =========================================================
    print("\n" + "=" * 80)
    print("[TEST 2] MazeChatPrompt (Stage B selector)")
    print("=" * 80)

    sel_msgs = selector_prompt.render_messages(sentence_prefix, word, candidates)
    sel_resp = sel_llm.invoke(sel_msgs)

    print("\n--- Raw Stage B output ---\n")
    print(sel_resp.content)

    maze_out = selector_prompt.parser.parse(sel_resp.content)

    print("\n--- Parsed MazeOutput ---")
    print("source      :", maze_out.source)
    print("distractor1 :", maze_out.distractor1)
    print("distractor2 :", maze_out.distractor2)
    print("distractor3 :", maze_out.distractor3)

    # =========================================================
    # 3) One-shot prompt (legacy)
    # =========================================================
    print("\n" + "=" * 80)
    print("[TEST 3] MazeLLMPrompt (one-shot)")
    print("=" * 80)

    oneshot_text = oneshot_prompt.render_text(sentence_prefix, word, candidates)
    print("\n--- Raw one-shot text ---\n")
    print(oneshot_text)

    oneshot_resp = sel_llm.invoke(oneshot_text)

    print("\n--- Raw one-shot output ---\n")
    print(oneshot_resp.content)

    oneshot_out = oneshot_prompt.parser.parse(oneshot_resp.content)

    print("\n--- Parsed one-shot MazeOutput ---")
    print("source      :", oneshot_out.source)
    print("distractor1 :", oneshot_out.distractor1)
    print("distractor2 :", oneshot_out.distractor2)
    print("distractor3 :", oneshot_out.distractor3)

    print("\n✅ All three prompts executed successfully.\n")

if __name__ == "__main__":
    main()
