"""Lockr (Keycloak) OIDC Authorization Code Flow."""

from __future__ import annotations

import urllib.parse
from datetime import datetime

import jwt
import requests
import streamlit as st

from auth.config import (
    OIDC_AUTH_URL,
    OIDC_CLIENT_ID,
    OIDC_CLIENT_SECRET,
    OIDC_LOGOUT_URL,
    OIDC_TOKEN_URL,
)
from auth.session import is_authenticated, set_user
from data_pipeline.db import get_client


def _get_redirect_uri() -> str:
    """Build redirect URI from the current Streamlit URL."""
    # In production, use env var; fallback to Streamlit's inferred URL
    import os
    override = os.getenv("OIDC_REDIRECT_URI")
    if override:
        return override
    # Streamlit >= 1.30 exposes the current URL
    try:
        ctx = st.context
        url = ctx.headers.get("Origin", "http://localhost:8501")
        return url.rstrip("/") + "/"
    except Exception:
        return "http://localhost:8501/"


def get_auth_url(state: str = "lockr") -> str:
    """Build Keycloak authorization URL."""
    params = {
        "client_id": OIDC_CLIENT_ID,
        "redirect_uri": _get_redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
    }
    return f"{OIDC_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str) -> dict | None:
    """Exchange authorization code for tokens.

    Returns dict with access_token, id_token, refresh_token or None on failure.
    """
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": _get_redirect_uri(),
        "client_id": OIDC_CLIENT_ID,
    }
    if OIDC_CLIENT_SECRET:
        data["client_secret"] = OIDC_CLIENT_SECRET

    try:
        resp = requests.post(OIDC_TOKEN_URL, data=data, timeout=15)
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def parse_id_token(id_token: str) -> dict | None:
    """Decode id_token JWT without signature verification (MVP).

    For production: verify signature against Keycloak JWKS.
    """
    try:
        payload = jwt.decode(
            id_token,
            options={"verify_signature": False},
            algorithms=["RS256"],
        )
        return payload
    except jwt.PyJWTError:
        return None


def _upsert_lockr_user(info: dict) -> None:
    """Create or update user record in app_users for Lockr login."""
    client = get_client()
    sub = info.get("sub", "")
    row = {
        "auth_type": "lockr",
        "first_name": info.get("given_name") or info.get("first_name") or info.get("username", ""),
        "last_name": info.get("family_name") or info.get("last_name") or "",
        "email": info.get("email"),
        "lockr_username": info.get("preferred_username") or info.get("username"),
        "lockr_sub": sub,
        "last_login": datetime.utcnow().isoformat(),
    }
    # Upsert by lockr_sub
    existing = (
        client.table("app_users")
        .select("id")
        .eq("lockr_sub", sub)
        .limit(1)
        .execute()
    )
    if existing.data:
        client.table("app_users").update(row).eq("lockr_sub", sub).execute()
    else:
        client.table("app_users").insert(row).execute()


def handle_callback() -> None:
    """Process OIDC callback — exchange code, parse token, set session.

    Call this early in app.py. If query params contain 'code' and 'state=lockr',
    this will authenticate the user and clear query params.
    """
    if is_authenticated():
        return

    params = st.query_params
    code = params.get("code")
    state = params.get("state")

    if not code or state != "lockr":
        return

    # Clear query params immediately to avoid re-processing
    st.query_params.clear()

    tokens = exchange_code(code)
    if not tokens:
        st.error("Не удалось получить токен от Lockr. Попробуйте снова.")
        return

    id_token = tokens.get("id_token", "")
    info = parse_id_token(id_token)
    if not info:
        st.error("Не удалось прочитать токен. Обратитесь к администратору.")
        return

    _upsert_lockr_user(info)

    set_user({
        "auth_type": "lockr",
        "first_name": info.get("given_name") or info.get("first_name") or info.get("username", ""),
        "last_name": info.get("family_name") or info.get("last_name") or "",
        "email": info.get("email"),
        "lockr_username": info.get("preferred_username") or info.get("username"),
    })
    st.rerun()


def get_logout_url(post_logout_redirect: str = "") -> str:
    """Build Keycloak logout URL."""
    params = {"client_id": OIDC_CLIENT_ID}
    if post_logout_redirect:
        params["post_logout_redirect_uri"] = post_logout_redirect
    return f"{OIDC_LOGOUT_URL}?{urllib.parse.urlencode(params)}"
