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
    """Show code input with countdown timer and resend button."""
    ts = st.session_state.get(_CODE_TS_KEY, 0)
    elapsed = int(time.time() - ts)
    remaining = max(0, _CODE_TTL - elapsed)
    mins, secs = divmod(remaining, 60)

    # Countdown + resend row
    col_timer, col_resend = st.columns([2, 1])
    with col_timer:
        if remaining > 0:
            st.markdown(
                f'<div style="color: #9ca3af; font-size: 0.85rem;">'
                f'Код действителен ещё <b style="color: #f0b429;">{mins}:{secs:02d}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="color: #ff6b6b; font-size: 0.85rem;">'
                'Код просрочен. Запросите новый.'
                '</div>',
                unsafe_allow_html=True,
            )
    with col_resend:
        resend_disabled = remaining > (_CODE_TTL - 30)  # allow resend after 30s
        if st.button(
            "Отправить повторно",
            key="_code_resend",
            disabled=resend_disabled,
        ):
            phone = st.session_state.get(_CODE_PHONE_KEY, "")
            if phone:
                chat_id = get_chat_id_by_phone(phone)
                if chat_id:
                    code = _generate_code()
                    _store_code(phone, code)
                    send_code(chat_id, code)
                    st.success("Новый код отправлен в Telegram.")
                    st.rerun()

    # Code input
    entered = st.text_input("Введите 6-значный код из Telegram", key="_code_input")
    if st.button("Подтвердить", key="_code_confirm", type="primary"):
        if _verify_code(entered):
            if is_registration:
                reg = st.session_state.pop("_ext_reg", {})
                _create_user(
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
            if remaining <= 0:
                st.error("Код просрочен. Нажмите «Отправить повторно».")
            else:
                st.error("Неверный код. Проверьте и попробуйте снова.")
