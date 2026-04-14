"""tests/unit/test_github_webhook.py — GitHub webhook route unit tests."""
from __future__ import annotations

import hashlib
import hmac
import json
import inspect
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.github_webhook import _verify_signature, router


# ── _verify_signature unit tests ─────────────────────────────────────────────

class TestVerifySignature:
    def _make_sig(self, payload: bytes, secret: str) -> str:
        return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    def test_valid_signature_returns_true(self):
        payload = b'{"action": "push"}'
        secret = "webhook-secret-abc"
        sig = self._make_sig(payload, secret)
        assert _verify_signature(payload, sig, secret) is True

    def test_wrong_secret_returns_false(self):
        payload = b'{"action": "push"}'
        sig = self._make_sig(payload, "correct-secret")
        assert _verify_signature(payload, sig, "wrong-secret") is False

    def test_empty_secret_returns_false(self):
        payload = b'{"action": "push"}'
        sig = self._make_sig(payload, "some-secret")
        assert _verify_signature(payload, sig, "") is False

    def test_tampered_payload_returns_false(self):
        secret = "webhook-secret"
        original = b'{"action": "push"}'
        sig = self._make_sig(original, secret)
        tampered = b'{"action": "push", "extra": "evil"}'
        assert _verify_signature(tampered, sig, secret) is False

    def test_empty_signature_returns_false(self):
        assert _verify_signature(b"payload", "", "secret") is False

    def test_constant_time_comparison_used(self):
        import api.routes.github_webhook as mod
        src = inspect.getsource(mod._verify_signature)
        assert "compare_digest" in src


# ── /webhook/github endpoint integration tests ───────────────────────────────

def _build_test_app():
    app = FastAPI()
    app.include_router(router)
    return app


def _sign(payload: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


class TestGithubWebhookEndpoint:
    def setup_method(self):
        self.app = _build_test_app()
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.secret = "test-webhook-secret-xyz"

    def test_missing_secret_returns_503(self):
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", ""):
            resp = self.client.post(
                "/webhook/github",
                content=b'{"ref": "refs/heads/main"}',
                headers={"x-github-event": "push", "x-hub-signature-256": "sha256=abc"},
            )
        assert resp.status_code == 503

    def test_invalid_signature_returns_401(self):
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=b'{"ref": "refs/heads/main"}',
                headers={
                    "x-github-event": "push",
                    "x-hub-signature-256": "sha256=badhex000",
                },
            )
        assert resp.status_code == 401

    def test_valid_push_event_accepted(self):
        payload = json.dumps({
            "ref": "refs/heads/main",
            "repository": {"clone_url": "https://github.com/acme/backend.git"},
            "commits": [{"id": "abc123", "modified": ["src/foo.py"]}],
        }).encode()
        sig = _sign(payload, self.secret)
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=payload,
                headers={"x-github-event": "push", "x-hub-signature-256": sig},
            )
        assert resp.status_code in (200, 202)

    def test_valid_pr_event_accepted(self):
        payload = json.dumps({
            "action": "opened",
            "pull_request": {"number": 42, "head": {"sha": "abc123"}},
            "repository": {"clone_url": "https://github.com/acme/backend.git"},
        }).encode()
        sig = _sign(payload, self.secret)
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=payload,
                headers={"x-github-event": "pull_request", "x-hub-signature-256": sig},
            )
        assert resp.status_code in (200, 202)

    def test_invalid_json_returns_400(self):
        payload = b"NOT-VALID-JSON"
        sig = _sign(payload, self.secret)
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=payload,
                headers={"x-github-event": "push", "x-hub-signature-256": sig},
            )
        assert resp.status_code == 400

    def test_push_without_repo_url_ignored(self):
        payload = json.dumps({
            "ref": "refs/heads/main",
            "repository": {},
            "commits": [],
        }).encode()
        sig = _sign(payload, self.secret)
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=payload,
                headers={"x-github-event": "push", "x-hub-signature-256": sig},
            )
        # Should respond 200 with "ignored" status, not crash
        assert resp.status_code in (200, 202)

    def test_ping_event_handled_gracefully(self):
        payload = json.dumps({"zen": "Design for failure."}).encode()
        sig = _sign(payload, self.secret)
        with patch("api.routes.github_webhook._WEBHOOK_SECRET", self.secret):
            resp = self.client.post(
                "/webhook/github",
                content=payload,
                headers={"x-github-event": "ping", "x-hub-signature-256": sig},
            )
        assert resp.status_code in (200, 202)
