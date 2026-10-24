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

FIX_MAX_ATTEMPTS = 3
MAX_CODE_BYTES = 200_000


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# Deployment settings. Locally the defaults work as they are.
# Browser origins allowed to call the API, comma separated (for example the Vercel URL).
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("EPOCHZERO_CORS_ORIGINS", "").split(",") if o.strip()] or [
    "http://localhost:3000", "http://127.0.0.1:3000"]
# Largest model the CPU dry run will build. Lower it on small hosts (Render free has 512 MB of memory).
DRY_RUN_MAX_PARAMS = _int_env("EPOCHZERO_DRY_RUN_MAX_PARAMS", 150_000_000)
# Checks (analyze calls) allowed per visitor IP per 10 minutes; 0 turns the limit off. A public deployment
# spends the Groq quota of whoever owns the key, so keep this on.
RATE_LIMIT_PER_10_MIN = _int_env("EPOCHZERO_RATE_LIMIT", 0)
# Sandbox timeouts. Small hosts have slow CPUs: importing torch can take over a minute, a dry run tens of seconds.
SANDBOX_TIMEOUT_SECONDS = _int_env("EPOCHZERO_SANDBOX_TIMEOUT", 15)
SANDBOX_STARTUP_SECONDS = _int_env("EPOCHZERO_SANDBOX_STARTUP", 180)
