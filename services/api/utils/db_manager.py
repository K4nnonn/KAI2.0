from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from azure.data.tables import TableServiceClient, UpdateMode
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
except Exception:  # pragma: no cover - optional dependency in some local setups
    TableServiceClient = None
    UpdateMode = None
    ResourceNotFoundError = Exception
    ResourceExistsError = Exception

APP_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = APP_ROOT / "kai_user_data.db"
_CHAT_TABLE_CLIENT = None
_CHAT_TABLE_ENABLED = os.environ.get("KAI_CHAT_HISTORY_TABLE_ENABLED", "true").lower() == "true"
_CHAT_TABLE_NAME = None


def _sanitize_table_name(name: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", name or "")
    if not cleaned:
        return "kaichathistory"
    if not cleaned[0].isalpha():
        cleaned = f"k{cleaned}"
    return cleaned[:63]


def _get_chat_table():
    """Return a cached Azure Table client for chat history when configured."""
    global _CHAT_TABLE_CLIENT, _CHAT_TABLE_NAME
    if not _CHAT_TABLE_ENABLED or TableServiceClient is None:
        return None
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return None
    if _CHAT_TABLE_NAME is None:
        _CHAT_TABLE_NAME = _sanitize_table_name(os.environ.get("KAI_CHAT_HISTORY_TABLE", "kai_chat_history"))
    if _CHAT_TABLE_CLIENT is not None:
        return _CHAT_TABLE_CLIENT
    try:
        service = TableServiceClient.from_connection_string(conn)
        table = service.get_table_client(_CHAT_TABLE_NAME)
        try:
            table.create_table()
        except ResourceExistsError:
            pass
        _CHAT_TABLE_CLIENT = table
        return table
    except Exception:
        return None


def _chat_row_key(ts_epoch: float) -> str:
    return f"{int(ts_epoch * 1000):013d}-{uuid4().hex}"


def _escape_odata(value: str) -> str:
    return value.replace("'", "''")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            role TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """
    )
    cur.execute("PRAGMA table_info(chat_history)")
    existing_cols = {row["name"] for row in cur.fetchall()}
    if "session_id" not in existing_cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_creatives (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP,
            payload TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_chat_message(role: str, message: str, session_id: str | None = None) -> None:
    session = (session_id or "").strip()
    if session:
        table = _get_chat_table()
        if table is not None and UpdateMode is not None:
            try:
                ts_epoch = time.time()
                entity = {
                    "PartitionKey": session,
                    "RowKey": _chat_row_key(ts_epoch),
                    "role": role,
                    "message": message,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_epoch)),
                    "ts_epoch": ts_epoch,
                }
                table.upsert_entity(mode=UpdateMode.REPLACE, entity=entity)
                return
            except Exception:
                pass
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_history (role, message, session_id) VALUES (?, ?, ?)",
        (role, message, session_id),
    )
    conn.commit()
    conn.close()


def fetch_chat_history(limit: int = 100, session_id: str | None = None) -> List[Dict[str, Any]]:
    session = (session_id or "").strip()
    if session:
        table = _get_chat_table()
        if table is not None:
            try:
                rows = []
                filt = f"PartitionKey eq '{_escape_odata(session)}'"
                entities = table.query_entities(
                    query_filter=filt,
                    select=["ts", "role", "message", "ts_epoch"],
                )
                for entity in entities:
                    rows.append(
                        {
                            "ts": entity.get("ts") or "",
                            "role": entity.get("role"),
                            "message": entity.get("message"),
                            "_ts_epoch": entity.get("ts_epoch") or 0,
                        }
                    )
                    if len(rows) >= max(200, limit):
                        break
                rows.sort(key=lambda r: r.get("_ts_epoch", 0))
                trimmed = rows[-limit:] if limit else rows
                for row in trimmed:
                    row.pop("_ts_epoch", None)
                return trimmed
            except Exception:
                pass
    conn = _get_conn()
    if session:
        cur = conn.execute(
            "SELECT ts, role, message FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session, limit),
        )
    else:
        cur = conn.execute(
            "SELECT ts, role, message FROM chat_history ORDER BY id DESC LIMIT ?", (limit,)
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return list(reversed(rows))


def clear_chat_history(session_id: str | None = None) -> None:
    session = (session_id or "").strip()
    if session:
        table = _get_chat_table()
        if table is not None:
            try:
                filt = f"PartitionKey eq '{_escape_odata(session)}'"
                entities = list(
                    table.query_entities(
                        query_filter=filt,
                        select=["PartitionKey", "RowKey"],
                    )
                )
                for entity in entities:
                    table.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])
            except Exception:
                pass
    conn = _get_conn()
    if session:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session,))
    else:
        conn.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()


def save_creative(payload: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute("INSERT INTO saved_creatives (payload) VALUES (?)", (json.dumps(payload),))
    conn.commit()
    conn.close()


def fetch_creatives(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.execute(
        "SELECT ts, payload FROM saved_creatives ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = []
    for r in cur.fetchall():
        try:
            data = json.loads(r["payload"])
        except Exception:
            data = {"raw": r["payload"]}
        data["ts"] = r["ts"]
        rows.append(data)
    conn.close()
    return rows


def set_api_key(name: str, value: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO api_keys (name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
        (name, value),
    )
    conn.commit()
    conn.close()


def get_api_key(name: str) -> Optional[str]:
    conn = _get_conn()
    cur = conn.execute("SELECT value FROM api_keys WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_setting(name: str, value: Any) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO settings (name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
        (name, json.dumps(value)),
    )
    conn.commit()
    conn.close()


def get_setting(name: str, default: Any = None) -> Any:
    conn = _get_conn()
    cur = conn.execute("SELECT value FROM settings WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]
