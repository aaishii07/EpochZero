"""HTTP layer: health without loading torch, CORS from the environment, and the per-visitor rate limit."""

import sys

from fastapi.testclient import TestClient

from app import config
from app.api import routes
from app.main import app

client = TestClient(app)


def test_health_does_not_import_torch_in_the_server():
    before = "torch" in sys.modules
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["torch"]
    if not before:
        assert "torch" not in sys.modules  # only the sandbox worker loads torch


def test_rate_limit_blocks_after_the_limit(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_10_MIN", 2)
    monkeypatch.setattr(config, "LLM_MODE", "mock")
    routes._hits.clear()
    headers = {"x-forwarded-for": "203.0.113.7"}
    body = {"code": "x = (1,\n"}  # a syntax error: fast and needs no LLM
    assert client.post("/analyze", json=body, headers=headers).status_code == 200
    assert client.post("/analyze", json=body, headers=headers).status_code == 200
    blocked = client.post("/analyze", json=body, headers=headers)
    assert blocked.status_code == 429 and "Try again" in blocked.json()["detail"]
    # Another visitor is not affected.
    assert client.post("/analyze", json=body, headers={"x-forwarded-for": "198.51.100.1"}).status_code == 200
    routes._hits.clear()


def test_stream_is_not_buffered_by_proxies(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODE", "mock")
    res = client.post("/analyze", json={"code": "x = (1,\n"})
    assert res.headers["x-accel-buffering"] == "no"
