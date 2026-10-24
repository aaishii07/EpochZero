import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.api.routes import router
from app.sandbox.runner import warm_up

app = FastAPI(title="EpochZero")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def _warm_sandbox():
    # Import torch in the sandbox worker now, in the background, so the first check is fast.
    threading.Thread(target=warm_up, daemon=True).start()
