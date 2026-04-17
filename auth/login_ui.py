"""Login page UI — simple external login or Lockr SSO."""

from __future__ import annotations

import streamlit as st

from auth.config import LOCKR_ENABLED
from auth.external_auth import render_login_form, render_registration_form


def render_login_page() -> None:
    """Full-page login screen shown when user is not authenticated."""
    st.markdown(
        '<div style="text-align: center; padding: 2rem 0 1rem;">'
        '<h1 style="color: #f0b429; font-size: 2.2rem;">MOEX Volume Analytics</h1>'
        '<p style="color: #9ca3af; font-size: 1rem;">Для доступа к дашборду необходима авторизация</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if LOCKR_ENABLED:
        tab_ext, tab_lockr = st.tabs(["Внешний пользователь", "Сотрудник Группы М"])
        with tab_ext:
            _render_external_tab()
        with tab_lockr:
            _render_lockr_tab()
    else:
        # Simplified auth: external users only
        _render_external_tab()


def _render_external_tab() -> None:
    """External user login/registration — name + phone only."""
    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #00d4ff; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.88rem; color: #d1d5db; line-height: 1.55;">'
        'Укажите имя, фамилию и номер телефона для доступа к дашборду. '
        'При повторном входе достаточно ввести номер телефона.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Выберите действие",
        ["Вход", "Регистрация"],
        horizontal=True,
        key="ext_auth_mode",
    )

    if mode == "Регистрация":
        render_registration_form()
    else:
        render_login_form()


def _render_lockr_tab() -> None:
    """Lockr SSO login — single button redirect."""
    from auth.lockr_oidc import get_auth_url

    st.markdown(
        '<div class="glass-card" style="border-left: 3px solid #f0b429; padding: 0.8rem 1rem;">'
        '<div style="font-size: 0.88rem; color: #d1d5db; line-height: 1.55;">'
        'Авторизация через корпоративную систему Lockr (Keycloak SSO).<br>'
        'Используйте доменный логин и пароль AD.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    auth_url = get_auth_url()
    st.markdown(
        f'<div style="text-align: center; padding: 2rem 0;">'
        f'<a href="{auth_url}" target="_self" '
        f'style="background: linear-gradient(135deg, #f0b429, #e09100); '
        f'color: #111; padding: 0.75rem 2.5rem; border-radius: 8px; '
        f'text-decoration: none; font-weight: 600; font-size: 1.1rem;">'
        f'Войти через Lockr SSO'
        f'</a></div>',
        unsafe_allow_html=True,
    )
