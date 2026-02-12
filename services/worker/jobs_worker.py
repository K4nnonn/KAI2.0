from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from utils.job_queue import (
    queue_client,
    get_job,
    update_job,
    store_job_result,
)
from utils.telemetry import log_event

import main as api


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _handle_job(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    job_type = job.get("jobType")
    payload = job.get("payload") or {}
    if job_type == "sa360_fetch":
        req = api.Sa360FetchRequest(**payload)
        return _run_async(api.sa360_fetch(req))
    if job_type == "sa360_fetch_and_audit":
        req = api.Sa360FetchRequest(**payload)
        return _run_async(api.sa360_fetch_and_audit(req))
    if job_type == "sa360_perf_window":
        req = api.Sa360DiagnosticsRequest(**payload)
        return _run_async(api.sa360_performance_diagnostics(req))
    if job_type == "sa360_plan_and_run":
        req = api.PlanRequest(**payload)
        return _run_async(api._chat_plan_and_run_core(req))
    if job_type == "audit_generate":
        req = api.AuditRequest(**payload)
        return _run_async(api.generate_audit(req))
    if job_type == "trends_seasonality":
        req = api.TrendsRequest(**payload)
        return _run_async(api.trends_seasonality(req))
    raise RuntimeError(f"Unknown job type: {job_type}")


def _normalize_result(result: Any) -> Any:
    if hasattr(result, "model_dump") and callable(result.model_dump):
        return result.model_dump()
    if hasattr(result, "dict") and callable(result.dict):
        return result.dict()
    return result


def _start_heartbeat(job_id: str, stage_ref: dict[str, str], interval: float) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()

    def _beat() -> None:
        while not stop_event.wait(interval):
            update_job(job_id, status="running", stage=stage_ref["value"], heartbeat=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event, thread


def _process_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        log_event("job_missing", job_id=job_id)
        return
    attempts = int(job.get("attempts") or 0) + 1
    stage_ref = {"value": "start"}
    update_job(job_id, status="running", attempts=attempts, stage=stage_ref["value"])
    log_event("job_start", job_id=job_id, job_type=job.get("jobType"), attempts=attempts, stage=stage_ref["value"])

    heartbeat_seconds = float(os.environ.get("JOB_HEARTBEAT_SECONDS", "30") or "30")
    stop_event = None
    heartbeat_thread = None
    if heartbeat_seconds > 0:
        stop_event, heartbeat_thread = _start_heartbeat(job_id, stage_ref, heartbeat_seconds)

    # Hard runtime guard to prevent indefinite stalls
    max_runtime_seconds = float(os.environ.get("JOB_MAX_RUNTIME_SECONDS", "0") or "0")
    watchdog_stop = threading.Event()

    def _watchdog() -> None:
        if max_runtime_seconds <= 0:
            return
        while not watchdog_stop.wait(5):
            elapsed = time.time() - start_ts
            if elapsed > max_runtime_seconds:
                try:
                    update_job(
                        job_id,
                        status="failed",
                        stage="timeout",
                        error=f"timeout after {max_runtime_seconds:.0f}s",
                    )
                except Exception:
                    pass
                log_event(
                    "job_timeout",
                    job_id=job_id,
                    job_type=job.get("jobType"),
                    elapsed_seconds=elapsed,
                    max_runtime_seconds=max_runtime_seconds,
                )
                os._exit(1)

    start_ts = time.time()
    watchdog_thread = None
    if max_runtime_seconds > 0:
        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

    previous_job_id = os.environ.get("KAI_JOB_ID")
    os.environ["KAI_JOB_ID"] = job_id
    try:
        stage_ref["value"] = "handle_job"
        update_job(job_id, status="running", stage=stage_ref["value"])
        log_event("job_stage", job_id=job_id, job_type=job.get("jobType"), stage=stage_ref["value"])
        result = _normalize_result(_handle_job(job_id, job))
        stage_ref["value"] = "store_result"
        update_job(job_id, status="running", stage=stage_ref["value"])
        log_event("job_stage", job_id=job_id, job_type=job.get("jobType"), stage=stage_ref["value"])
        result_blob = store_job_result(job_id, result)
        update_job(job_id, status="succeeded", resultBlob=result_blob, stage="succeeded")
        log_event("job_success", job_id=job_id, job_type=job.get("jobType"))
    except Exception as exc:
        tb = traceback.format_exc()
        tb_short = tb[-4000:] if tb and len(tb) > 4000 else tb
        error_type = type(exc).__name__
        max_attempts = int(os.environ.get("JOB_MAX_ATTEMPTS", "3") or "3")
        if attempts < max_attempts:
            update_job(job_id, status="queued", error=str(exc)[:500], stage="retry")
            queue_client().send_message(job_id)
            log_event(
                "job_retry",
                job_id=job_id,
                job_type=job.get("jobType"),
                error=str(exc),
                error_type=error_type,
                traceback=tb_short,
            )
        else:
            update_job(job_id, status="failed", error=str(exc)[:2000], stage="failed")
            log_event(
                "job_failed",
                job_id=job_id,
                job_type=job.get("jobType"),
                error=str(exc),
                error_type=error_type,
                traceback=tb_short,
            )
    finally:
        watchdog_stop.set()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1.0)
        if stop_event is not None:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=heartbeat_seconds)
        if previous_job_id is None:
            os.environ.pop("KAI_JOB_ID", None)
        else:
            os.environ["KAI_JOB_ID"] = previous_job_id


def main() -> None:
    poll_seconds = float(os.environ.get("JOB_QUEUE_POLL_SECONDS", "2") or "2")
    visibility_timeout = int(os.environ.get("JOB_QUEUE_VISIBILITY_TIMEOUT", "300") or "300")
    _start_health_server()
    log_event("job_worker_start", poll_seconds=poll_seconds, visibility_timeout=visibility_timeout)

    q = queue_client()
    while True:
        try:
            messages = q.receive_messages(messages_per_page=1, visibility_timeout=visibility_timeout)
            for msg in messages:
                job_id = msg.content
                _process_job(job_id)
                try:
                    q.delete_message(msg.id, msg.pop_receipt)
                except Exception:
                    pass
        except Exception as exc:
            log_event("job_worker_error", error=str(exc))
            time.sleep(max(1.0, poll_seconds))
            continue

        time.sleep(max(0.5, poll_seconds))


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):  # noqa: A003 - match BaseHTTPRequestHandler signature
        return


def _start_health_server() -> None:
    port = int(os.environ.get("PORT", "8000") or "8000")
    server = _ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


if __name__ == "__main__":
    main()
