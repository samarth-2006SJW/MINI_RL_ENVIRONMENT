"""
server/app.py — OpenEnv multi-mode deployment entry point.
Required by openenv validate (server.app:main in pyproject.toml).
Thin wrapper that mounts the Gradio UI onto the FastAPI app and serves it.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import gradio as gr
from app import app_api, demo


def main():
    """Entry point for openenv serve / multi-mode deployment."""
    application = gr.mount_gradio_app(app_api, demo, path="/")
    uvicorn.run(application, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
