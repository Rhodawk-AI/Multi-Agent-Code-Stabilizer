from __future__ import annotations

import logging
import os
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])

_IS_DEV = os.environ.get("RHODAWK_ENV", "production").lower() == "development"


class TokenRequest(BaseModel):
    client_id: str
    client_secret: str
    scopes: list[str] | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str | None = None


@router.post("/auth/token")
async def issue_token(req: TokenRequest) -> TokenResponse:
    _client_id = os.environ.get("RHODAWK_CLIENT_ID", "")
    _client_secret = os.environ.get("RHODAWK_CLIENT_SECRET", "")

    if not _client_id or not _client_secret:
        if _IS_DEV and os.environ.get("RHODAWK_DEV_AUTH") == "1":
            from auth.jwt_middleware import create_access_token, create_refresh_token
            scopes = req.scopes or [
                "runs:read", "runs:write", "issues:read", "fixes:read",
                "escalations:read", "escalations:write",
            ]
            token = create_access_token(sub=req.client_id, scopes=scopes)
            refresh = create_refresh_token(sub=req.client_id)
            return TokenResponse(
                access_token=token,
                expires_in=3600,
                refresh_token=refresh,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth not configured. Set RHODAWK_CLIENT_ID and RHODAWK_CLIENT_SECRET.",
        )

    if req.client_id != _client_id or req.client_secret != _client_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials.",
        )

    from auth.jwt_middleware import create_access_token, create_refresh_token

    scopes = req.scopes or [
        "runs:read", "runs:write", "issues:read", "fixes:read",
        "escalations:read", "escalations:write",
    ]
    token = create_access_token(sub=req.client_id, scopes=scopes)
    refresh = create_refresh_token(sub=req.client_id)
    return TokenResponse(
        access_token=token,
        expires_in=3600,
        refresh_token=refresh,
    )
