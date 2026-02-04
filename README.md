# Multilingual A-Maze Task Enhanced by LLM Agent


This project is a multilingual version of the A-Maze Task, enhanced by an LLM agent. It generates multiple distractors for each word in a sentence using an LLM and a lexicon or multiple LLMs. It produces JSON outputs that can be parsed to build maze tasks.

## Quick start
1. Configure `config.yaml` with your model and vocabulary paths.
2. Run `api/llm_agent.py` to generate outputs with open-source LLMs.
3. Run `api/chat_agent.py` to generate outputs with OpenAI's Chat API.
