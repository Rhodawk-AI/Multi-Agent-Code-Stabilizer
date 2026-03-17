from __future__ import annotations

import hashlib
import hmac
import json
import logging

from brain.schemas import AuditTrailEntry

log = logging.getLogger(__name__)

_DEFAULT_SECRET = "openmoss-default-insecure-key-change-in-production"


class AuditTrailSigner:

    def __init__(self, secret: str = "") -> None:
        self._secret = (secret or _DEFAULT_SECRET).encode("utf-8")
        if not secret:
            log.warning(
                "AuditTrailSigner: no HMAC secret provided. "
                "Set hmac_secret in StabilizerConfig for production use."
            )

    def sign(self, entry: AuditTrailEntry) -> str:
        payload = json.dumps({
            "id": entry.id,
            "run_id": entry.run_id,
            "event_type": entry.event_type,
            "entity_id": entry.entity_id,
            "entity_type": entry.entity_type,
            "before_state": entry.before_state,
            "after_state": entry.after_state,
            "actor": entry.actor,
            "timestamp": entry.timestamp.isoformat(),
        }, sort_keys=True).encode("utf-8")

        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def verify(self, entry: AuditTrailEntry) -> bool:
        if not entry.hmac_signature:
            return False
        expected = self.sign(entry)
        return hmac.compare_digest(expected, entry.hmac_signature)

    def verify_chain(self, entries: list[AuditTrailEntry]) -> list[str]:
        failures: list[str] = []
        for entry in entries:
            if not self.verify(entry):
                failures.append(
                    f"Audit trail integrity failure: entry {entry.id} "
                    f"(event={entry.event_type}, ts={entry.timestamp.isoformat()}) "
                    "signature mismatch — possible tampering"
                )
        return failures
