from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import threading

from app.api.routes import router
from app.sandbox.runner import warm_up

app = FastAPI(title="EpochZero")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def _warm_sandbox():
    # Import torch in the sandbox worker now, in the background, so the first check is fast.
    threading.Thread(target=warm_up, daemon=True).start()
