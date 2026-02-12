from __future__ import annotations

import streamlit as st
from pathlib import Path


def load_custom_css() -> None:
    """Inject global CSS (style.css) and chrome cleanup."""
    app_root = Path(__file__).resolve().parents[2]
    css_path = app_root / "style.css"
    css_text = ""
    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
    css = f"""
    <style>
    {css_text}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
