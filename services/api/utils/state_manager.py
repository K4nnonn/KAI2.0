from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# Ensure app root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Source-based engine import with fallback to compiled wheel
try:
    from kai_core.core_logic import UnifiedAuditEngine  # type: ignore
    print("LOADED: Source Code Engine (kai_core.core_logic)")
except ImportError as e1:
    try:
        import kai_core_engine  # type: ignore
        from kai_core_engine import UnifiedAuditEngine  # type: ignore
        print("LOADED: Compiled Binary Engine (kai_core_engine)")
    except ImportError as e2:
        UnifiedAuditEngine = None  # type: ignore
        print(f"WARNING: No engine available. Source import error: {e1}. Wheel import error: {e2}")


def init_session_state() -> None:
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
    if UnifiedAuditEngine is None:
        return None
    template_path = ROOT / "kai_core" / "GenerateAudit" / "template.xlsx"
    if not template_path.exists():
        # Fallback to azure path for backwards compatibility
        template_path = ROOT / "azure" / "GenerateAudit" / "template.xlsx"
    data_dir = Path(tempfile.mkdtemp())
    return UnifiedAuditEngine(
        template_path=template_path,
        data_directory=data_dir,
        business_unit="Kai",
        business_context={},
    )
