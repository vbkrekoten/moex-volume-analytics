"""External user registration/login — simplified: name + phone only.

Telegram verification is temporarily disabled. Users enter their name
and phone number, and the app creates/updates a user record directly.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from auth.session import set_user
from data_pipeline.db import get_client


def _normalize_phone(raw: str) -> str:
    """Normalize phone to +7XXXXXXXXXX format."""
    digits = "".join(c for c in raw if c.isdigit())
    if digits.startswith("8") and len(digits) == 11:
        digits = "7" + digits[1:]
    if not digits.startswith("7"):
        digits = "7" + digits
    return f"+{digits}"


def _is_valid_phone(phone: str) -> bool:
    """Check normalized phone matches +7XXXXXXXXXX (11 digits total)."""
    digits = phone.lstrip("+")
    return digits.isdigit() and len(digits) == 11


# --- DB helpers ---

def _find_user_by_phone(phone: str) -> dict | None:
    client = get_client()
    resp = client.table("app_users").select("*").eq("phone", phone).limit(1).execute()
    return resp.data[0] if resp.data else None


def _create_user(first_name: str, last_name: str, phone: str, email: str = "") -> dict:
    client = get_client()
    row = {
        "auth_type": "external",
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "email": email or None,
        "phone_verified": False,  # Telegram verification disabled
        "last_login": datetime.utcnow().isoformat(),
    }
    resp = client.table("app_users").insert(row).execute()
    return resp.data[0]


def _update_last_login(phone: str) -> None:
    client = get_client()
    client.table("app_users").update(
        {"last_login": datetime.utcnow().isoformat()}
    ).eq("phone", phone).execute()


# --- Public API ---

def render_registration_form() -> None:
    """Render registration form for new users (name + phone, no verification)."""
    st.markdown("##### Регистрация")
    with st.form("ext_register"):
        first_name = st.text_input("Имя *")
        last_name = st.text_input("Фамилия *")
        phone_raw = st.text_input("Телефон * (+7...)")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Зарегистрироваться", type="primary")

    if not submitted:
        return

    if not first_name or not last_name or not phone_raw:
        st.error("Заполните обязательные поля: имя, фамилия, телефон.")
        return

    phone = _normalize_phone(phone_raw)
    if not _is_valid_phone(phone):
        st.error("Некорректный номер телефона. Ожидаемый формат: +7XXXXXXXXXX.")
        return

    existing = _find_user_by_phone(phone)
    if existing:
        st.warning("Этот номер уже зарегистрирован. Используйте «Вход».")
        return

    _create_user(first_name.strip(), last_name.strip(), phone, email.strip())
    set_user({
        "auth_type": "external",
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "email": email.strip() or None,
        "phone": phone,
    })
    st.rerun()


def render_login_form() -> None:
    """Render login form for existing users (phone only)."""
    st.markdown("##### Вход по телефону")
    with st.form("ext_login"):
        phone_raw = st.text_input("Телефон (+7...)")
        submitted = st.form_submit_button("Войти", type="primary")

    if not submitted:
        return

    if not phone_raw:
        st.error("Введите номер телефона.")
        return

    phone = _normalize_phone(phone_raw)
    if not _is_valid_phone(phone):
        st.error("Некорректный номер телефона. Ожидаемый формат: +7XXXXXXXXXX.")
        return

    existing = _find_user_by_phone(phone)
    if not existing:
        st.error("Номер не найден. Пройдите регистрацию.")
        return

    _update_last_login(phone)
    set_user({
        "auth_type": "external",
        "first_name": existing.get("first_name", ""),
        "last_name": existing.get("last_name", ""),
        "email": existing.get("email"),
        "phone": phone,
    })
    st.rerun()
