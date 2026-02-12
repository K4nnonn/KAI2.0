from __future__ import annotations

import contextvars
import json
from datetime import datetime
from typing import Any

_request_id_var = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    return _request_id_var.get()


def log_event(event: str, **fields: Any) -> None:
    record = {"event": event, "ts": datetime.utcnow().isoformat() + "Z"}
    request_id = get_request_id()
    if request_id:
        record["request_id"] = request_id
    record.update(fields)
    try:
        print(json.dumps(record, default=str), flush=True)
    except Exception:
        pass
