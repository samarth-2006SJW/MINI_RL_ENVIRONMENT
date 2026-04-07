import os
import json
import time
import gradio as gr
import openai
from environment import WarehouseEnvironment
from models import WarehouseState

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
def _get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    base_url = os.getenv("LLM_BASE_URL", None)
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return openai.OpenAI(**client_kwargs)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")