from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from typing import Any

try:
    from azure.data.tables import TableServiceClient  # type: ignore
    from azure.storage.blob import BlobServiceClient  # type: ignore
    from azure.storage.queue import QueueClient  # type: ignore
    _AZURE_QUEUE_AVAILABLE = True
except Exception:
    TableServiceClient = None  # type: ignore[assignment]
    BlobServiceClient = None  # type: ignore[assignment]
    QueueClient = None  # type: ignore[assignment]
    _AZURE_QUEUE_AVAILABLE = False


_QUEUE_CLIENT: QueueClient | None = None
_TABLE_CLIENT = None
_BLOB_CLIENT = None


def job_queue_enabled() -> bool:
    return os.environ.get("JOB_QUEUE_ENABLED", "false").lower() == "true"


def job_queue_force() -> bool:
    return os.environ.get("JOB_QUEUE_FORCE", "false").lower() == "true"


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_connection_string() -> str:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not configured")
    return conn


def _normalize_queue_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (name or "").lower()).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if len(cleaned) < 3:
        cleaned = fallback
    return cleaned[:63]


def _normalize_container(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (name or "").lower()).strip("-")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    if len(cleaned) < 3:
        cleaned = fallback
    return cleaned[:63]


def _queue_client() -> QueueClient:
    global _QUEUE_CLIENT
    if not _AZURE_QUEUE_AVAILABLE or QueueClient is None:
        raise RuntimeError("Azure queue client not available in this runtime")
    if _QUEUE_CLIENT is not None:
        return _QUEUE_CLIENT
    name = _normalize_queue_name(os.environ.get("JOB_QUEUE_NAME", "kai-jobs"), "kai-jobs")
    client = QueueClient.from_connection_string(_get_connection_string(), queue_name=name)
    try:
        client.create_queue()
    except Exception:
        pass
    _QUEUE_CLIENT = client
    return client


def queue_client() -> QueueClient:
    return _queue_client()


def _table_client():
    global _TABLE_CLIENT
    if not _AZURE_QUEUE_AVAILABLE or TableServiceClient is None:
        raise RuntimeError("Azure table client not available in this runtime")
    if _TABLE_CLIENT is not None:
        return _TABLE_CLIENT
    table_name = os.environ.get("JOB_TABLE_NAME", "kaiJobs")
    service = TableServiceClient.from_connection_string(_get_connection_string())
    client = service.get_table_client(table_name)
    try:
        client.create_table()
    except Exception:
        pass
    _TABLE_CLIENT = client
    return client


def _result_container():
    global _BLOB_CLIENT
    if not _AZURE_QUEUE_AVAILABLE or BlobServiceClient is None:
        raise RuntimeError("Azure blob client not available in this runtime")
    if _BLOB_CLIENT is not None:
        return _BLOB_CLIENT
    container = _normalize_container(os.environ.get("JOB_RESULT_CONTAINER", "kai-job-results"), "kai-job-results")
    service = BlobServiceClient.from_connection_string(_get_connection_string())
    client = service.get_container_client(container)
    try:
        client.create_container()
    except Exception:
        pass
    _BLOB_CLIENT = client
    return client


def enqueue_job(job_type: str, payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    now = _utc_iso()
    entity = {
        "PartitionKey": "job",
        "RowKey": job_id,
        "status": "queued",
        "jobType": job_type,
        "createdAt": now,
        "updatedAt": now,
        "attempts": 0,
        "payload": json.dumps(payload or {}),
    }
    table = _table_client()
    table.upsert_entity(entity)
    queue = _queue_client()
    queue.send_message(job_id)
    return job_id


def get_job(job_id: str) -> dict[str, Any] | None:
    table = _table_client()
    try:
        entity = table.get_entity(partition_key="job", row_key=job_id)
    except Exception:
        return None
    payload_raw = entity.get("payload")
    payload = {}
    if isinstance(payload_raw, str) and payload_raw:
        try:
            payload = json.loads(payload_raw)
        except Exception:
            payload = {}
    entity["payload"] = payload
    return entity


def update_job(job_id: str, **fields: Any) -> None:
    table = _table_client()
    entity = {
        "PartitionKey": "job",
        "RowKey": job_id,
        "updatedAt": _utc_iso(),
    }
    entity.update(fields)
    table.upsert_entity(entity, mode="merge")


def store_job_result(job_id: str, result: Any) -> str | None:
    try:
        payload = json.dumps(result, default=str).encode("utf-8")
    except Exception:
        return None
    blob_name = f"{job_id}.json"
    container = _result_container()
    container.upload_blob(name=blob_name, data=payload, overwrite=True)
    return blob_name


def get_job_result(job_id: str) -> dict[str, Any] | None:
    job = get_job(job_id)
    if not job:
        return None
    blob_name = job.get("resultBlob")
    if not blob_name:
        return None
    container = _result_container()
    try:
        data = container.download_blob(blob_name).readall()
        return json.loads(data)
    except Exception:
        return None


def queue_depth() -> int | None:
    try:
        props = _queue_client().get_queue_properties()
        return int(props.approximate_message_count)
    except Exception:
        return None
