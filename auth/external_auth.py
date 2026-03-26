"""External user registration and login via phone + Telegram verification."""

from __future__ import annotations

import random
import time
from datetime import datetime

import streamlit as st

from auth.session import set_user
from auth.telegram_bot import send_code, get_chat_id_by_phone
from data_pipeline.db import get_client


_CODE_KEY = "_auth_code"
_CODE_TS_KEY = "_auth_code_ts"
_CODE_PHONE_KEY = "_auth_code_phone"
_CODE_TTL = 300  # 5 minutes


def _generate_code() -> str:
    return f"{random.randint(100000, 999999)}"


def _store_code(phone: str, code: str) -> None:
    st.session_state[_CODE_KEY] = code
    st.session_state[_CODE_TS_KEY] = time.time()
    st.session_state[_CODE_PHONE_KEY] = phone


def _verify_code(entered: str) -> bool:
    stored = st.session_state.get(_CODE_KEY)
    ts = st.session_state.get(_CODE_TS_KEY, 0)
    if not stored:
        return False
    if time.time() - ts > _CODE_TTL:
        return False
    return entered.strip() == stored


def _clear_code() -> None:
    for k in (_CODE_KEY, _CODE_TS_KEY, _CODE_PHONE_KEY):
        st.session_state.pop(k, None)


def _normalize_phone(raw: str) -> str:
    """Normalize phone to +7XXXXXXXXXX format."""
    digits = "".join(c for c in raw if c.isdigit())
    if digits.startswith("8") and len(digits) == 11:
        digits = "7" + digits[1:]
    if not digits.startswith("7"):
        digits = "7" + digits
    return f"+{digits}"


# --- DB helpers ---

def _find_user_by_phone(phone: str) -> dict | None:
    client = get_client()
    resp = client.table("app_users").select("*").eq("phone", phone).limit(1).execute()
    return resp.data[0] if resp.data else None


def _create_user(first_name: str, last_name: str, phone: str, email: str) -> dict:
    client = get_client()
    row = {
        "auth_type": "external",
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "email": email or None,
        "phone_verified": True,
        "last_login": datetime.utcnow().isoformat(),
    }
    resp = client.table("app_users").insert(row).execute()
    return resp.data[0]


def _update_last_login(phone: str) -> None:
    client = get_client()
    client.table("app_users").update(
        {"last_login": datetime.utcnow().isoformat(), "phone_verified": True}
    ).eq("phone", phone).execute()


# --- Public API ---

def render_registration_form() -> None:
    """Render registration form for new external users."""
    st.markdown("##### Регистрация")
    with st.form("ext_register"):
        first_name = st.text_input("Имя *")
        last_name = st.text_input("Фамилия *")
        phone_raw = st.text_input("Телефон * (+7...)")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Получить код в Telegram")

    if submitted:
        if not first_name or not last_name or not phone_raw:
            st.error("Заполните обязательные поля.")
            return

        phone = _normalize_phone(phone_raw)
        existing = _find_user_by_phone(phone)
        if existing:
            st.warning("Этот номер уже зарегистрирован. Используйте «Вход».")
            return

        chat_id = get_chat_id_by_phone(phone)
        if not chat_id:
            from auth.config import TG_BOT_USERNAME
            bot_link = f"https://t.me/{TG_BOT_USERNAME}" if TG_BOT_USERNAME else "Telegram-бот"
            st.error(
                f"Номер {phone} не привязан к Telegram-боту. "
                f"Сначала напишите боту {bot_link} и поделитесь номером телефона."
            )
            return

        code = _generate_code()
        _store_code(phone, code)
        ok = send_code(chat_id, code)
        if ok:
            st.session_state["_ext_reg"] = {
                "first_name": first_name, "last_name": last_name,
                "phone": phone, "email": email,
            }
            st.success("Код отправлен в Telegram. Введите его ниже.")
        else:
            st.error("Не удалось отправить код. Попробуйте позже.")

    # Code verification
    if st.session_state.get(_CODE_KEY) and st.session_state.get("_ext_reg"):
        _render_code_input(is_registration=True)


def render_login_form() -> None:
    """Render login form for returning external users."""
    st.markdown("##### Вход по телефону")
    with st.form("ext_login"):
        phone_raw = st.text_input("Телефон (+7...)")
        submitted = st.form_submit_button("Получить код в Telegram")

    if submitted and phone_raw:
        phone = _normalize_phone(phone_raw)
        existing = _find_user_by_phone(phone)
        if not existing:
            st.error("Номер не найден. Пройдите регистрацию.")
            return

        chat_id = get_chat_id_by_phone(phone)
        if not chat_id:
            st.error("Номер не привязан к Telegram-боту.")
            return

        code = _generate_code()
        _store_code(phone, code)
        ok = send_code(chat_id, code)
        if ok:
            st.session_state["_ext_login_phone"] = phone
            st.success("Код отправлен в Telegram.")
        else:
            st.error("Не удалось отправить код.")

    # Code verification
    if st.session_state.get(_CODE_KEY) and st.session_state.get("_ext_login_phone"):
        _render_code_input(is_registration=False)


def _render_code_input(is_registration: bool) -> None:
    """Show code input and verify."""
    entered = st.text_input("Введите 6-значный код из Telegram", key="_code_input")
    if st.button("Подтвердить", key="_code_confirm"):
        if _verify_code(entered):
            if is_registration:
                reg = st.session_state.pop("_ext_reg", {})
                user = _create_user(
                    reg["first_name"], reg["last_name"],
                    reg["phone"], reg.get("email", ""),
                )
                set_user({
                    "auth_type": "external",
                    "first_name": reg["first_name"],
                    "last_name": reg["last_name"],
                    "email": reg.get("email"),
                    "phone": reg["phone"],
                })
            else:
                phone = st.session_state.pop("_ext_login_phone", "")
                _update_last_login(phone)
                existing = _find_user_by_phone(phone)
                set_user({
                    "auth_type": "external",
                    "first_name": existing["first_name"],
                    "last_name": existing["last_name"],
                    "email": existing.get("email"),
                    "phone": phone,
                })
            _clear_code()
            st.rerun()
        else:
            st.error("Неверный или просроченный код.")
