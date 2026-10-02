
# The Multilingual Agentic Maze Task


This project is a multilingual version of the A-Maze Task, enhanced by an LLM agent. It generates multiple distractors for each word in a sentence using an LLM and a lexicon or multiple LLMs. It produces JSON outputs that can be parsed to build maze tasks.

## Quick start
1. Edit `config.yaml` (usually just `LANGUAGE_CODE`, `MODEL_ID`, `PROCESSING_MODE`, and `USE_LEXICON_CANDIDATES`).
2. Run `api/llm_agent.py` for local Hugging Face models.
3. Run `api/chat_agent.py` for OpenAI chat models.

Both CLIs share **`--use-lexicon-candidates` / `--no-use-lexicon-candidates`** (same semantics as `USE_LEXICON_CANDIDATES` in YAML). If you omit the flag on `chat_agent.py`, it follows `config.yaml`; previously `chat_agent.py` defaulted to non-lexicon unless explicitly enabled.

### Auto path resolution
If `LEXICON_PATH`, `INPUT_DATA_PATH`, `TEMPLATE_DIR`, or `OUTPUT_PATH` are empty in `config.yaml`,
the app resolves them automatically from `LANGUAGE_CODE`:

- Lexicon: `data/lexicon/lexicon_{language_code}.txt`
- Input (`naturalistic_reading`): `data/input/{language_code}/naturalistic_{language_code}.txt`
- Input (`controlled_experiment`): `data/input/{language_code}/controlled_{language_code}.txt`
- Template dir: `template/{language_code}`
- Output: `results/{language_code}/{agent}_out_{lang}_{model}.json`

You can still override paths from CLI (`--input-data-path`, `--lexicon-path`, `--template-dir`, `--output-path`).

## Command examples
Run commands from the repository root:

```bash
cd /path/to/Agentic-Maze   # use your local clone path
```

Chinese Tab-separated snippet example file: `data/input/zh/test/Ins_HumanRights_zh.txt` — pass **`--word-separator $'\t'`** so columns split correctly.

### 1) LLM agent (`api/llm_agent.py`): lexicon candidate pools

Uses YAML default **`USE_LEXICON_CANDIDATES`** (typically `true`) unless overridden.

```bash
python api/llm_agent.py \
  --config-path config.yaml \
  --language-code zh \
  --processing-mode naturalistic_reading \
  --use-lexicon-candidates \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --input-data-path data/input/zh/multipleye/Ins_HumanRights_zh.txt \
  --word-separator $'\t' \
  --output-path results/zh/multipleye/llm_lexicon_Ins_HumanRights_zh.json \
  --limit 100
```

### 2) LLM agent: LM-proposed candidates (no lexicon neighborhoods)

```bash
python api/llm_agent.py \
  --config-path config.yaml \
  --language-code zh \
  --processing-mode naturalistic_reading \
  --no-use-lexicon-candidates \
  --input-data-path data/input/zh/test/Ins_HumanRights_zh.txt \
  --word-separator $'\t' \
  --output-path results/zh/test_hf_lm_candidates_ins.json \
  --limit 8
```

### 3) LLM agent: different HF models for proposal vs selection

Candidate pools still follow **`USE_LEXICON_CANDIDATES`** from config unless you add **`--no-use-lexicon-candidates`**.

```bash
python api/llm_agent.py \
  --config-path config.yaml \
  --language-code zh \
  --input-data-path data/input/zh/test/Ins_HumanRights_zh.txt \
  --word-separator $'\t' \
  --proposal-model-id Qwen/Qwen3-4B-Instruct-2507 \
  --selection-model-id Qwen/Qwen2.5-3B-Instruct \
  --output-path results/zh/test_hf_dual_models_ins.json \
  --limit 8
```

### 4) Chat agent (`api/chat_agent.py`): lexicon candidate pools

```bash
python api/chat_agent.py \
  --config-path config.yaml \
  --language-code en \
  --processing-mode controlled_experiment \
  --use-lexicon-candidates \
  --model-id gpt-4o-mini \
  --input-data-path data/input/en/controlled_en.txt \
  --word-separator " " \
  --output-path results/en/test_controlled_chat.json \
  --limit 10
```

### 5) Chat agent: LM-proposed candidates (`--no-use-lexicon-candidates`)

```bash
python api/chat_agent.py \
  --config-path config.yaml \
  --language-code zh \
  --no-use-lexicon-candidates \
  --model-id gpt-4o-mini \
  --input-data-path data/input/zh/test/Ins_HumanRights_zh.txt \
  --word-separator $'\t' \
  --output-path results/zh/test_chat_no_lexicon_ins.json \
  --limit 8
```

### 6) Chat agent: different OpenAI models for proposal vs selection

```bash
python api/chat_agent.py \
  --config-path config.yaml \
  --language-code zh \
  --no-use-lexicon-candidates \
  --model-id gpt-4o-mini \
  --proposal-model-id gpt-5-mini \
  --selection-model-id gpt-4o-mini \
  --input-data-path data/input/zh/test/Ins_HumanRights_zh.txt \
  --word-separator $'\t' \
  --output-path results/zh/test_chat_dual_openai_ins.json \
  --limit 8
```

## Web app
Run a browser UI for sentence/file input, settings, JSON preview, and download:

```bash
python web_app.py --share
```

- `--share` creates a public Gradio link.
- Without `--share`, open the local URL (default `http://0.0.0.0:7860`).

## Free public hosting (Hugging Face Spaces)
You can deploy the app for free and share a stable URL:

```bash
gradio deploy --provider spaces --app-file app.py --title "Agentic A-Maze Studio"
```

After deployment:
- Your Space URL stays stable.
- When you push code updates to the Space repo, the app rebuilds and the same URL shows the new version.

Notes:
- `app.py` is the Spaces entrypoint and serves `web_app.build_app()`.
- `requirements.txt` is used by Spaces for dependency install.
