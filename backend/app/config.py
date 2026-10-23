"""Runtime settings read from the environment (backend/.env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# The fixer writes code, so it gets the stronger model. The goal reader and planner make short decisions, so they
# use a faster model. Groq rate limits are per model, so splitting also spreads the free-tier token budget.
MODEL = os.getenv("EPOCHZERO_MODEL", "").strip() or "openai/gpt-oss-120b"
FAST_MODEL = os.getenv("EPOCHZERO_FAST_MODEL", "").strip() or "openai/gpt-oss-20b"
# Longest wait for a rate-limit reset before falling back to offline mode for that call.
MAX_RATE_LIMIT_WAIT_SECONDS = 8
LLM_MODE = (os.getenv("EPOCHZERO_LLM", "").strip() or "live").lower()  # "live" or "mock"
if LLM_MODE == "live" and not GROQ_API_KEY:
    LLM_MODE = "mock"

PLANNER_MAX_TOOL_CALLS = 6
PLANNER_MAX_SECONDS = 60
FIX_MAX_ATTEMPTS = 3
SANDBOX_TIMEOUT_SECONDS = 15
MAX_CODE_BYTES = 200_000
MAX_PATCH_LINES = 15
