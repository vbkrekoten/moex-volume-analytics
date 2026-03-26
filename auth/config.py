"""Auth configuration — reads from env vars → st.secrets → Supabase app_config."""

import os

_cache: dict[str, str] = {}


def _load_from_supabase() -> dict[str, str]:
    """Load all config keys from Supabase app_config table (cached)."""
    if _cache:
        return _cache
    try:
        from data_pipeline.db import get_client
        client = get_client()
        resp = client.table("app_config").select("key,value").execute()
        for row in resp.data or []:
            _cache[row["key"]] = row["value"]
    except Exception:
        pass
    return _cache


def _get(key: str, default: str = "") -> str:
    """Read config: env var → st.secrets → Supabase app_config → default."""
    # 1. Environment variable
    val = os.getenv(key)
    if val:
        return val
    # 2. Streamlit secrets
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    # 3. Supabase app_config table
    db_conf = _load_from_supabase()
    if key in db_conf:
        return db_conf[key]
    return default


def _bool(val: str | None) -> bool:
    return str(val).lower() in ("1", "true", "yes")


AUTH_ENABLED: bool = _bool(_get("AUTH_ENABLED", "false"))

# Lockr / Keycloak OIDC settings
OIDC_SERVER_URL: str = _get("OIDC_SERVER_URL", "https://sso-ldap.beta.moex.com/auth")
OIDC_REALM: str = _get("OIDC_REALM", "LDAP")
OIDC_CLIENT_ID: str = _get("OIDC_CLIENT_ID", "moex-volume-analytics")
OIDC_CLIENT_SECRET: str = _get("OIDC_CLIENT_SECRET")

# Telegram bot for phone verification
TG_BOT_TOKEN: str = _get("TG_BOT_TOKEN")
TG_BOT_USERNAME: str = _get("TG_BOT_USERNAME")

# Feature flags
LOCKR_ENABLED: bool = bool(OIDC_CLIENT_SECRET)

# Derived OIDC endpoints
OIDC_BASE = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect"
OIDC_AUTH_URL = f"{OIDC_BASE}/auth"
OIDC_TOKEN_URL = f"{OIDC_BASE}/token"
OIDC_USERINFO_URL = f"{OIDC_BASE}/userinfo"
OIDC_LOGOUT_URL = f"{OIDC_BASE}/logout"
OIDC_JWKS_URL = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect/certs"
