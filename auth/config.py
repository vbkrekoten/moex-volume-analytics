"""Auth configuration — reads from env vars with st.secrets fallback."""

import os


def _get(key: str, default: str = "") -> str:
    """Read from env var first, then st.secrets (Streamlit Cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, default))
    except Exception:
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
LOCKR_ENABLED: bool = bool(OIDC_CLIENT_SECRET)  # Lockr tab visible only if configured

# Derived OIDC endpoints
OIDC_BASE = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect"
OIDC_AUTH_URL = f"{OIDC_BASE}/auth"
OIDC_TOKEN_URL = f"{OIDC_BASE}/token"
OIDC_USERINFO_URL = f"{OIDC_BASE}/userinfo"
OIDC_LOGOUT_URL = f"{OIDC_BASE}/logout"
OIDC_JWKS_URL = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect/certs"
