"""Session management via st.session_state."""

from __future__ import annotations

import streamlit as st


_KEY = "auth_user"


def is_authenticated() -> bool:
    return _KEY in st.session_state and st.session_state[_KEY] is not None


def get_current_user() -> dict | None:
    return st.session_state.get(_KEY)


def set_user(user: dict) -> None:
    """Store user dict in session.

    Expected keys: auth_type, first_name, last_name, email, phone,
    lockr_username (optional).
    """
    st.session_state[_KEY] = user


def clear_session() -> None:
    st.session_state.pop(_KEY, None)
