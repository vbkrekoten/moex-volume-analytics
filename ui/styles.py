"""Custom CSS for glassmorphism dark theme."""

import streamlit as st


def load_custom_css():
    """Inject custom CSS for glassmorphism styling."""
    st.markdown("""
    <style>
    /* === Glassmorphism cards === */
    .glass-card {
        background: rgba(17, 24, 39, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(240, 180, 41, 0.12);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.3s;
    }
    .glass-card:hover {
        border-color: rgba(240, 180, 41, 0.3);
    }

    /* === Section headers with gradient === */
    .section-header {
        background: linear-gradient(135deg, rgba(240,180,41,0.12) 0%, rgba(0,212,255,0.08) 100%);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 1.8rem 0 1rem 0;
        border-left: 3px solid #f0b429;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f0e6d3;
    }
    .section-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        color: #9ca3af;
    }

    /* === KPI metric styling === */
    [data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.45);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(240, 180, 41, 0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }

    /* === Factor card === */
    .factor-card {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        min-height: 120px;
        transition: all 0.3s;
    }
    .factor-card:hover {
        border-color: rgba(240, 180, 41, 0.25);
        transform: translateY(-1px);
    }
    .factor-card .factor-name {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .factor-card .factor-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .factor-card .factor-detail {
        font-size: 0.7rem;
        color: #6b7280;
    }

    /* Traffic light colors */
    .status-green { color: #51cf66; }
    .status-yellow { color: #ffa94d; }
    .status-red { color: #ff6b6b; }

    /* === Smooth scroll === */
    html { scroll-behavior: smooth; }

    /* === Divider === */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(240,180,41,0.2), transparent);
        margin: 2rem 0;
    }

    /* === Popover styling === */
    .formula-block {
        background: rgba(10, 14, 23, 0.8);
        border: 1px solid rgba(240, 180, 41, 0.15);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        color: #f0b429;
    }

    /* Hide default streamlit header padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Sidebar nav links */
    .nav-link {
        display: block;
        padding: 0.3rem 0.5rem;
        color: #9ca3af;
        text-decoration: none;
        border-radius: 6px;
        margin-bottom: 0.2rem;
        font-size: 0.85rem;
    }
    .nav-link:hover {
        background: rgba(240, 180, 41, 0.1);
        color: #f0b429;
    }
    </style>
    """, unsafe_allow_html=True)
