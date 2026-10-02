from __future__ import annotations

from dotenv import load_dotenv

from web_app import build_app

load_dotenv()
demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

