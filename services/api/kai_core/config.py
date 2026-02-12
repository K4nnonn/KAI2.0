"""
Connector-level configuration helpers.

Purposefully lightweight: keeps the core engine untouched while letting
the connector adapt to LOCAL vs ENTERPRISE environments via env flags.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Literal

DeploymentMode = Literal["LOCAL", "ENTERPRISE"]

_ROOT = Path(__file__).resolve().parents[1]
_SETTINGS_PATH = _ROOT / "settings.json"


def _load_settings() -> dict:
    defaults = {
        "system_prompt_override": "",
        "force_enterprise_mode": False,
        "custom_data_path": "",
        "debug_logging": False,
        "tenant_id_override": "",
        "maintenance_mode": False,
    }
    try:
        if _SETTINGS_PATH.exists():
            loaded = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            defaults.update({k: loaded.get(k, v) for k, v in defaults.items()})
    except Exception:
        # Fallback to defaults if settings.json is malformed or unreadable.
        pass
    return defaults


_SETTINGS = _load_settings()


def get_deployment_mode() -> DeploymentMode:
    """
    Return the active deployment mode.
    Defaults to LOCAL to preserve upstream developer behavior when the flag is absent.
    """
    raw_env = (os.environ.get("KAI_DEPLOYMENT_MODE") or "").strip().upper()
    if raw_env in {"LOCAL", "ENTERPRISE"}:
        return "ENTERPRISE" if raw_env == "ENTERPRISE" else "LOCAL"

    # Fallback to settings.json override if no env var is set.
    if _SETTINGS.get("force_enterprise_mode"):
        return "ENTERPRISE"
    return "LOCAL"


def is_enterprise() -> bool:
    return get_deployment_mode() == "ENTERPRISE"


def get_settings() -> dict:
    return dict(_SETTINGS)


def get_custom_data_path() -> Path | None:
    custom = (_SETTINGS.get("custom_data_path") or "").strip()
    if custom:
        return Path(custom)
    return None


def get_system_prompt_override() -> str:
    return (_SETTINGS.get("system_prompt_override") or "").strip()


def is_debug_logging() -> bool:
    return bool(_SETTINGS.get("debug_logging"))


def get_tenant_override() -> str:
    return (_SETTINGS.get("tenant_id_override") or "").strip()


def is_maintenance_mode() -> bool:
    return bool(_SETTINGS.get("maintenance_mode"))


_BOOL_TRUE = {"1", "true", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "no", "n", "off"}


def _bool_env(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    return default


def is_cost_saving_mode() -> bool:
    return bool(_bool_env("KAI_COST_SAVING_MODE", False))


def is_azure_openai_enabled() -> bool:
    override = _bool_env("AZURE_OPENAI_ENABLED")
    if override is not None:
        return override
    disabled = _bool_env("AZURE_OPENAI_DISABLED")
    if disabled is not None:
        return not disabled
    # Default to local-only in LOCAL mode unless explicitly enabled.
    if get_deployment_mode() == "LOCAL":
        return False
    if is_cost_saving_mode():
        return False
    return True


def is_azure_embeddings_enabled() -> bool:
    override = _bool_env("AZURE_OPENAI_EMBEDDINGS_ENABLED")
    if override is not None:
        return override
    if is_cost_saving_mode():
        return False
    return is_azure_openai_enabled()
