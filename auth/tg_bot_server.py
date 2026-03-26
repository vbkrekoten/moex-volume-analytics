"""Standalone Telegram bot — links phone numbers to chat IDs for verification.

Run as: python -m auth.tg_bot_server

The bot asks users to share their phone number, then stores the
phone ↔ chat_id mapping in Supabase `app_users` table.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOKEN = os.getenv("TG_BOT_TOKEN", "")
API = f"https://api.telegram.org/bot{TOKEN}"


def _send(chat_id: int, text: str, reply_markup: dict | None = None) -> None:
    payload: dict = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = json.dumps(reply_markup)
    requests.post(f"{API}/sendMessage", data=payload, timeout=10)


def _request_phone(chat_id: int) -> None:
    markup = {
        "keyboard": [[{"text": "Поделиться номером телефона", "request_contact": True}]],
        "resize_keyboard": True,
        "one_time_keyboard": True,
    }
    _send(
        chat_id,
        "Для привязки аккаунта MOEX Volume Analytics нажмите кнопку ниже "
        "и поделитесь номером телефона.",
        reply_markup=markup,
    )


def _normalize_phone(raw: str) -> str:
    digits = "".join(c for c in raw if c.isdigit())
    if digits.startswith("8") and len(digits) == 11:
        digits = "7" + digits[1:]
    if not digits.startswith("7"):
        digits = "7" + digits
    return f"+{digits}"


def _save_chat_id(phone: str, chat_id: int) -> None:
    """Upsert phone ↔ chat_id in app_users."""
    from data_pipeline.db import get_client

    client = get_client()
    # Check if user exists by phone
    resp = client.table("app_users").select("id").eq("phone", phone).limit(1).execute()
    if resp.data:
        client.table("app_users").update(
            {"telegram_chat_id": chat_id}
        ).eq("phone", phone).execute()
        log.info("Updated chat_id for %s -> %d", phone, chat_id)
    else:
        # Create a placeholder user — will be completed during registration
        client.table("app_users").insert({
            "auth_type": "external",
            "first_name": "—",
            "last_name": "—",
            "phone": phone,
            "telegram_chat_id": chat_id,
            "phone_verified": False,
        }).execute()
        log.info("Created placeholder user for %s -> %d", phone, chat_id)


def _process_update(update: dict) -> None:
    msg = update.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    if not chat_id:
        return

    # /start command
    text = msg.get("text", "")
    if text.startswith("/start"):
        _request_phone(chat_id)
        return

    # Contact shared
    contact = msg.get("contact")
    if contact:
        phone_raw = contact.get("phone_number", "")
        phone = _normalize_phone(phone_raw)
        _save_chat_id(phone, chat_id)
        _send(
            chat_id,
            f"Номер {phone} привязан. Теперь вы можете войти в MOEX Volume Analytics.",
        )
        return


def main() -> None:
    if not TOKEN:
        log.error("TG_BOT_TOKEN not set. Exiting.")
        sys.exit(1)

    log.info("Starting Telegram bot (polling)...")
    offset = 0
    while True:
        try:
            resp = requests.get(
                f"{API}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            if not resp.ok:
                log.warning("getUpdates returned %d", resp.status_code)
                time.sleep(5)
                continue

            data = resp.json()
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                _process_update(upd)

        except requests.RequestException as e:
            log.warning("Polling error: %s", e)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Bot stopped.")
            break


if __name__ == "__main__":
    main()
