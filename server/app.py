"""
Server entry point for OpenEnv multi-mode deployment.
Thin wrapper that imports and runs the main FastAPI + Gradio app.
"""
import sys
import os

# Add the project root to sys.path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import gradio as gr
from app import app_api, demo


def main():
    """Entry point for `openenv serve` / multi-mode deployment."""
    app = gr.mount_gradio_app(app_api, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
