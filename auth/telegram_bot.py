"""Telegram Bot API helper — send verification codes."""

from __future__ import annotations

import requests

from auth.config import TG_BOT_TOKEN


def send_code(chat_id: int, code: str) -> bool:
    """Send a 6-digit verification code to a Telegram chat.

    Returns True if the message was sent successfully.
    """
    if not TG_BOT_TOKEN or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": (
            f"Код подтверждения для MOEX Volume Analytics:\n\n"
            f"**{code}**\n\n"
            f"Код действителен 5 минут."
        ),
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.ok
    except requests.RequestException:
        return False


def get_chat_id_by_phone(phone: str) -> int | None:
    """Look up Telegram chat_id for a phone number in app_users.

    Returns chat_id or None if not found / not linked.
    """
    from data_pipeline.db import get_client

    client = get_client()
    resp = (
        client.table("app_users")
        .select("telegram_chat_id")
        .eq("phone", phone)
        .not_.is_("telegram_chat_id", "null")
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0]["telegram_chat_id"]
    return None
