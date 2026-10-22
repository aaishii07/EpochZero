"""Runtime settings read from the environment (backend/.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL = os.getenv("EPOCHZERO_MODEL", "claude-opus-5-5")
LLM_MODE = os.getenv("EPOCHZERO_LLM", "live").lower()  # "live" or "mock"
if LLM_MODE == "live" and not os.getenv("ANTHROPIC_API_KEY"):
    LLM_MODE = "mock"

PLANNER_MAX_TOOL_CALLS = 6
PLANNER_MAX_SECONDS = 60
FIX_MAX_ATTEMPTS = 3
SANDBOX_TIMEOUT_SECONDS = 15
MAX_CODE_BYTES = 200_000
MAX_PATCH_LINES = 15
