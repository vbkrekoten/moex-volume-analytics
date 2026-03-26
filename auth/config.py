"""Auth configuration — reads from environment variables."""

import os


def _bool(val: str | None) -> bool:
    return str(val).lower() in ("1", "true", "yes")


AUTH_ENABLED: bool = _bool(os.getenv("AUTH_ENABLED", "false"))

# Lockr / Keycloak OIDC settings
OIDC_SERVER_URL: str = os.getenv("OIDC_SERVER_URL", "https://sso-ldap.beta.moex.com/auth")
OIDC_REALM: str = os.getenv("OIDC_REALM", "LDAP")
OIDC_CLIENT_ID: str = os.getenv("OIDC_CLIENT_ID", "moex-volume-analytics")
OIDC_CLIENT_SECRET: str = os.getenv("OIDC_CLIENT_SECRET", "")

# Telegram bot for phone verification
TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN", "")
TG_BOT_USERNAME: str = os.getenv("TG_BOT_USERNAME", "")  # e.g. "moex_auth_bot"

# Derived OIDC endpoints
OIDC_BASE = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect"
OIDC_AUTH_URL = f"{OIDC_BASE}/auth"
OIDC_TOKEN_URL = f"{OIDC_BASE}/token"
OIDC_USERINFO_URL = f"{OIDC_BASE}/userinfo"
OIDC_LOGOUT_URL = f"{OIDC_BASE}/logout"
OIDC_JWKS_URL = f"{OIDC_SERVER_URL}/realms/{OIDC_REALM}/protocol/openid-connect/certs"
