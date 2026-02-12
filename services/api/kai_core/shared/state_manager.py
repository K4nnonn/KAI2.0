"""
Shared session state and resource caching for the Streamlit MPA.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Try to import from source code first, then fall back to wheel
try:
    from kai_core.core_logic import UnifiedAuditEngine  # type: ignore
except ImportError:
    # Fall back to wheel if source import fails
    ROOT = Path(__file__).resolve().parents[2]
    DIST_LIBS = ROOT / "dist" / "libs"
    if DIST_LIBS.exists() and str(DIST_LIBS) not in sys.path:
        sys.path.append(str(DIST_LIBS))
    try:
        from kai_core_engine import UnifiedAuditEngine  # type: ignore
    except ImportError:
        UnifiedAuditEngine = None  # type: ignore


def init_session_state() -> None:
    """
    Ensure required session keys exist across pages.
    """
    defaults: Dict[str, Any] = {
        "user_id": "guest",
        "uploaded_data": None,
        "audit_results": None,
        "pmax_results": None,
        "creative_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def get_engine() -> Optional["UnifiedAuditEngine"]:
    """
    Load the core engine once for all pages. If the wheel is not available,
    return None so the UI can stay functional (with limited features).
    """
    if UnifiedAuditEngine is None:
        return None
    root = Path(__file__).resolve().parents[2]
    template_path = root / "azure" / "GenerateAudit" / "template.xlsx"
    data_dir = Path(tempfile.mkdtemp())
    return UnifiedAuditEngine(
        template_path=template_path,
        data_directory=data_dir,
        business_unit="Kai",
        business_context={},
    )
