#!/usr/bin/env python3
"""
Song Cover Pipeline — Web UI Server

FastAPI backend providing REST API + WebSocket progress for the
song cover pipeline.  Designed for internal network deployment.

Usage:
    cd webui && python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote, unquote

# ---------------------------------------------------------------------------
# Path setup -- project root is the parent of this webui/ directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webui")


class _TaskLogHandler(logging.Handler):
    """
    A logging handler that captures records for a single task.

    Writes to both an in-memory buffer (for API retrieval) and a
    per-task log file on disk.  Attached temporarily during pipeline
    execution and removed afterward.
    """

    def __init__(self, log_path: Path):
        super().__init__()
        self.log_path = log_path
        self._records: List[logging.LogRecord] = []
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)
        try:
            msg = self.format(record)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            self.handleError(record)

    def get_log_text(self) -> str:
        """Return all captured log lines as a single string."""
        lines = []
        for r in self._records:
            try:
                lines.append(self.format(r))
            except Exception:
                lines.append(str(r.msg))
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEBUI_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = WEBUI_DIR / "uploads"
DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "ria_cover.yaml"
MODELS_DIR = _PROJECT_ROOT / "models" / "DDSP"

# Upload / storage limits
MAX_UPLOAD_SIZE_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB max per file
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024               # 8 MB write chunks
MAX_BATCH_FILES = 50                               # max files per batch job

# ---- Tiered cleanup strategy ----
# After a successful task, keep only these stage directories;
# all others are deleted immediately to save disk space.
KEEP_STAGES_ON_SUCCESS = ["04_final_mix"]   # only the final cover
# Optionally keep converted dry vocals: ["03_timbre_conversion", "04_final_mix"]

# Retention: how long before entire task directories are deleted.
# Intermediates of successful tasks are cleaned immediately, so these
# timeouts mainly affect failed tasks and opted-in "keep" tasks.
INTERMEDIATE_RETENTION_HOURS = 1   # delete non-final stages after 1 h
TASK_RETENTION_HOURS = 24          # delete entire task dir after 24 h
FAILED_RETENTION_HOURS = 6         # delete failed task dirs after 6 h

# Disk pressure: when free space drops below this threshold, start
# aggressively deleting oldest completed tasks (newest first).
DISK_PRESSURE_FREE_GB = 20
MAX_TOTAL_UPLOADS_GB = 100

# Supported input formats
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv", ".m4v"}
ALLOWED_EXTS = AUDIO_EXTS | VIDEO_EXTS

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaskState:
    """In-memory state for one pipeline task."""
    id: str
    status: str = "pending"          # pending | running | completed | failed
    progress: float = 0.0            # 0-100
    stage: str = ""                  # current stage key
    message: str = ""                # human-readable
    output_files: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    input_filename: str = ""
    model_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


@dataclass
class BatchFileEntry:
    """State for one file inside a batch job."""
    file_id: str                        # == upload_id == task_id of the sub-task
    input_filename: str
    input_path: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)
    config: Dict[str, Any] = field(default_factory=dict)
    model_ckpt: str = ""
    keep_intermediates: bool = False
    status: str = "pending"             # pending | running | completed | failed
    progress: float = 0.0
    stage: str = ""
    message: str = ""
    error: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)


@dataclass
class BatchState:
    """In-memory state for a batch job."""
    id: str
    status: str = "pending"             # pending | running | completed | failed
    progress: float = 0.0
    current_index: int = 0
    message: str = ""
    error: Optional[str] = None
    files: List[BatchFileEntry] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Task Manager (thread-safe, single-task)
# ---------------------------------------------------------------------------

class TaskManager:
    """
    Manages pipeline task lifecycle.

    Only ONE task can run at a time (GPU is shared).  Additional
    submissions while a task is running receive HTTP 409.

    Thread-safety:
      - Task state reads are lock-free (single-writer pattern).
      - Lock ordering: _lock -> _tasks dict mutation.
      - ``_running_task_id`` tracks which task currently holds the GPU,
        and is ONLY modified while _lock is held.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskState] = {}
        self._ws_subscribers: Dict[str, List[WebSocket]] = {}
        self._running_task_id: Optional[str] = None   # None if idle (only valid under lock)
        # Batch tracking
        self._batches: Dict[str, BatchState] = {}
        self._batch_subscribers: Dict[str, List[WebSocket]] = {}

    # -- Public queries (lock-free reads -- safe since dict writes are GIL-atomic) --

    def is_running(self) -> bool:
        """True if any task is currently executing (GPU is busy)."""
        return self._running_task_id is not None

    def create_task(self, input_filename: str, model_name: str) -> TaskState:
        task_id = uuid.uuid4().hex[:12]
        task = TaskState(
            id=task_id,
            input_filename=input_filename,
            model_name=model_name,
        )
        self._tasks[task_id] = task
        self._ws_subscribers.setdefault(task_id, [])
        return task

    def get_task(self, task_id: str) -> Optional[TaskState]:
        return self._tasks.get(task_id)

    # -- WebSocket subscriber management --

    def subscribe(self, task_id: str, ws: WebSocket) -> None:
        """Register a WebSocket for progress updates; auto-creates entry if missing."""
        if task_id not in self._ws_subscribers:
            self._ws_subscribers[task_id] = []
        self._ws_subscribers[task_id].append(ws)

    def unsubscribe(self, task_id: str, ws: WebSocket) -> None:
        if task_id in self._ws_subscribers:
            subs = self._ws_subscribers[task_id]
            if ws in subs:
                subs.remove(ws)

    async def broadcast(self, task_id: str, data: dict) -> None:
        """Push a JSON message to all WebSocket subscribers for a task."""
        subs = self._ws_subscribers.get(task_id, [])
        if not subs:
            return
        dead: List[WebSocket] = []
        payload = json.dumps(data)
        for ws in subs:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unsubscribe(task_id, ws)

    def update_task(self, task_id: str, **kwargs) -> None:
        """Thread-safe task state update + schedule async broadcast."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)

        # Build progress payload
        data: Dict[str, Any] = {
            "type": "progress",
            "task_id": task.id,
            "status": task.status,
            "progress": task.progress,
            "stage": task.stage,
            "message": task.message,
        }
        if task.output_files:
            data["output_files"] = task.output_files
        if task.error:
            data["error"] = task.error

        # Schedule broadcast on the event loop (thread-safe)
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda d=data: asyncio.ensure_future(self.broadcast(task_id, d))
            )
        except RuntimeError:
            pass  # no running event loop (e.g. during import)

    # -- Batch management --

    def create_batch(self, files_meta: List[Dict[str, Any]]) -> BatchState:
        """Create a BatchState and register sub-TaskStates for each file."""
        batch_id = uuid.uuid4().hex[:12]
        batch = BatchState(id=batch_id)
        for meta in files_meta:
            upload_id = meta["upload_id"]
            input_path = meta["input_path"]
            output_dir = meta["output_dir"]
            config = meta["config"]
            model_ckpt = meta.get("model_ckpt", "")
            input_filename = meta.get("input_filename", input_path.name)
            keep_intermediates = meta.get("keep_intermediates", False)

            # Create sub-TaskState
            task = TaskState(
                id=upload_id,
                input_filename=input_filename,
                model_name=Path(model_ckpt).stem if model_ckpt else "",
            )
            self._tasks[upload_id] = task
            self._ws_subscribers.setdefault(upload_id, [])

            entry = BatchFileEntry(
                file_id=upload_id,
                input_filename=input_filename,
                input_path=input_path,
                output_dir=output_dir,
                config=config,
                model_ckpt=model_ckpt,
                keep_intermediates=keep_intermediates,
            )
            batch.files.append(entry)

        self._batches[batch_id] = batch
        self._batch_subscribers.setdefault(batch_id, [])
        return batch

    def get_batch(self, batch_id: str) -> Optional[BatchState]:
        return self._batches.get(batch_id)

    def subscribe_batch(self, batch_id: str, ws: WebSocket) -> None:
        if batch_id not in self._batch_subscribers:
            self._batch_subscribers[batch_id] = []
        self._batch_subscribers[batch_id].append(ws)

    def unsubscribe_batch(self, batch_id: str, ws: WebSocket) -> None:
        if batch_id in self._batch_subscribers:
            subs = self._batch_subscribers[batch_id]
            if ws in subs:
                subs.remove(ws)

    async def broadcast_batch(self, batch_id: str, data: dict) -> None:
        subs = self._batch_subscribers.get(batch_id, [])
        if not subs:
            return
        dead: List[WebSocket] = []
        payload = json.dumps(data)
        for ws in subs:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unsubscribe_batch(batch_id, ws)

    def _batch_payload(self, batch: BatchState) -> Dict[str, Any]:
        """Serialize a batch snapshot for WS/REST."""
        files_data = []
        for f in batch.files:
            fd: Dict[str, Any] = {
                "file_id": f.file_id,
                "input_filename": f.input_filename,
                "status": f.status,
                "progress": f.progress,
                "stage": f.stage,
                "message": f.message,
                "error": f.error,
            }
            if f.output_files:
                fd["output_files"] = _available_output_files(f.file_id, f.output_files)
            else:
                fd["output_files"] = None
            files_data.append(fd)

        return {
            "type": "batch_progress",
            "batch_id": batch.id,
            "status": batch.status,
            "progress": batch.progress,
            "current_index": batch.current_index,
            "message": batch.message,
            "error": batch.error,
            "files": files_data,
        }

    def _broadcast_batch_state(self, batch: BatchState) -> None:
        """Schedule a batch broadcast on the event loop (thread-safe)."""
        payload = self._batch_payload(batch)
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda d=payload: asyncio.ensure_future(self.broadcast_batch(batch.id, d))
            )
        except RuntimeError:
            pass

    def _batch_report(self, batch: BatchState, entry: BatchFileEntry,
                      idx: int, kwargs: Dict[str, Any]) -> None:
        """Bridge between _execute_single and batch state.

        Updates the sub-TaskState, mirrors state to the BatchFileEntry,
        recomputes batch progress, and broadcasts the batch snapshot.
        """
        # 1. Update sub-TaskState (keeps per-file endpoints in sync)
        self.update_task(entry.file_id, **kwargs)

        # 2. Mirror to BatchFileEntry
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)

        # 3. Recompute batch progress
        N = max(len(batch.files), 1)
        pct = entry.progress
        if entry.status in ("completed", "failed"):
            batch.progress = round(((idx + 1) / N) * 100, 1)
        else:
            batch.progress = round(((idx + pct / 100.0) / N) * 100, 1)

        batch.current_index = idx
        batch.status = "running"
        batch.message = kwargs.get("message", batch.message)

        # 4. Broadcast batch snapshot
        self._broadcast_batch_state(batch)

    # -- Pipeline execution core (lock-free) --

    def _execute_single(
        self,
        task_id: str,
        config: dict,
        input_path: Path,
        output_dir: Path,
        model_ckpt: str,
        keep_intermediates: bool,
        report: Callable[..., None],
    ) -> str:
        """
        Run ONE pipeline to completion.  Lock-free -- the caller holds the GPU lock.

        ``report`` is invoked at every state change with the same keyword args
        as update_task (status, progress, stage, message, error, output_files).

        Returns the final status: 'completed' or 'failed'.
        """
        task = self._tasks.get(task_id)

        # Mark started
        if task is not None:
            task.started_at = datetime.now().isoformat()
        report(status="running", progress=0.0, stage="starting",
               message="Initializing pipeline...")

        from src.pipeline import SongCoverPipeline

        # Progress dispatcher: handles BOTH pipeline stage callbacks
        # and DDSP segment callbacks.
        def _dispatcher(*args):
            if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
                # DDSP segment-level: (current, total)
                current, total = args
                pct = 48.0 + (current / max(total, 1)) * 40.0
                pct = min(pct, 88.0)
                report(
                    status="running",
                    progress=round(pct, 1),
                    stage="timbre_conversion",
                    message=f"DDSP inference: segment {current}/{total}",
                )
            elif len(args) >= 4:
                # Pipeline stage-level: (stage, status, percent, message)
                stage, status, pct, msg = args[0], args[1], args[2], args[3]
                if status == "error":
                    current_task = self._tasks.get(task_id)
                    report(
                        status="failed",
                        progress=current_task.progress if current_task else 0.0,
                        stage=stage,
                        message=msg,
                    )
                else:
                    current_task = self._tasks.get(task_id)
                    report(
                        status="running",
                        progress=pct if pct >= 0 else (current_task.progress if current_task else 0.0),
                        stage=stage,
                        message=msg,
                    )

        # Build and run the pipeline
        _saved_cwd = os.getcwd()
        os.chdir(str(_PROJECT_ROOT))

        # Per-task file logging
        task_dir = output_dir.parent  # uploads/<task_id>
        log_handler = _TaskLogHandler(task_dir / "task.log")
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        stage_timings: Dict[str, Dict[str, Any]] = {}
        _stage_start_time = time.time()

        final_status = "completed"
        try:
            pipeline = SongCoverPipeline(config)

            # Web tasks always start fresh (no resume)
            if pipeline._checkpoint_file.is_file():
                pipeline._checkpoint_file.unlink()
                pipeline._completed_stages = set()

            # Wrap the dispatcher to track per-stage timing
            def _dispatcher_with_timing(*args):
                nonlocal _stage_start_time
                if len(args) >= 4:
                    stage, status = args[0], args[1]
                    if status == "started":
                        _stage_start_time = time.time()
                        stage_timings[stage] = {"start": _stage_start_time}
                    elif status in ("completed", "error") and stage in stage_timings:
                        stage_timings[stage]["end"] = time.time()
                        stage_timings[stage]["elapsed"] = round(
                            time.time() - stage_timings[stage]["start"], 1
                        )
                _dispatcher(*args)

            pipeline.run(progress_callback=_dispatcher_with_timing)
        except Exception as exc:
            logger.exception("Task %s failed", task_id)
            final_status = "failed"
            report(
                status="failed",
                error=str(exc),
                message=f"Error: {exc}",
            )
            # Save structured error result
            try:
                error_result = {
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(exc),
                    "started_at": task.started_at if task else None,
                    "finished_at": datetime.now().isoformat(),
                    "log_file": str(task_dir / "task.log"),
                }
                with open(task_dir / "task_result.json", "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass
        else:
            # ---- Save structured task result ----
            output_files: Dict[str, str] = {}
            try:
                input_stem = Path(config["task"]["input_song"]).stem
                cover_rel = Path("04_final_mix") / f"{input_stem}_cover.wav"
                cover_abs = output_dir / cover_rel
                if cover_abs.is_file():
                    output_files["cover"] = str(cover_rel)

                for stage_dir_name, label in [
                    ("03_timbre_conversion", "converted_vocals"),
                    ("01_harmony_separation", "vocals"),
                    ("01_harmony_separation", "instrumental"),
                    ("02_reverb_separation", "noreverb"),
                    ("02_reverb_separation", "reverb"),
                ]:
                    stage_dir = output_dir / stage_dir_name
                    if stage_dir.is_dir():
                        for wav in sorted(stage_dir.glob("*.wav")):
                            if label not in output_files:
                                output_files[label] = str(wav.relative_to(output_dir))
                            break

                # Clean up first, then publish the file list.  Otherwise the
                # UI receives links to intermediate files that are deleted
                # immediately afterward when ``keep_intermediates`` is off.
                if not keep_intermediates:
                    try:
                        n = _cleanup_task_intermediates(
                            output_dir.parent,
                            KEEP_STAGES_ON_SUCCESS,
                        )
                        if n > 0:
                            logger.info("Task %s: cleaned %d intermediate stage(s), kept %s",
                                        task_id, n, KEEP_STAGES_ON_SUCCESS)
                    except Exception:
                        pass  # cleanup failure is non-fatal

                # Only expose files that still exist after cleanup.  This also
                # handles a partially produced stage without creating 404 links.
                output_files = {
                    key: rel_path
                    for key, rel_path in output_files.items()
                    if (output_dir / rel_path).is_file()
                }

                task_result = {
                    "task_id": task_id,
                    "task_name": config["task"].get("name", "unnamed"),
                    "model_ckpt": model_ckpt,
                    "input_file": str(input_path),
                    "output_dir": str(output_dir),
                    "status": "completed",
                    "stages": stage_timings,
                    "output_files": {k: str(v) for k, v in output_files.items()},
                    "started_at": task.started_at if task else None,
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_total_s": round(
                        sum(s.get("elapsed", 0) for s in stage_timings.values()), 1
                    ),
                    "log_file": str(task_dir / "task.log"),
                }
                with open(task_dir / "task_result.json", "w", encoding="utf-8") as f:
                    json.dump(task_result, f, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass  # non-fatal

            # Report completion with the final, downloadable file list.
            report(
                status="completed",
                progress=100.0,
                stage="mixing",
                message="Pipeline complete! Your cover is ready.",
                output_files=output_files,
            )

        finally:
            root_logger.removeHandler(log_handler)
            os.chdir(_saved_cwd)
            if task:
                task.finished_at = datetime.now().isoformat()

        return final_status

    # -- Pipeline execution wrappers --

    def run_task(
        self,
        task_id: str,
        config: dict,
        input_path: Path,
        output_dir: Path,
        model_ckpt: str,
        keep_intermediates: bool = False,
    ) -> None:
        """
        Run a single pipeline task synchronously in a background thread.

        Acquires the task lock; if another task is already running the
        current task is immediately marked 'failed' with HTTP 409 semantics.
        """
        # ---- Acquire the single-task lock ----
        if not self._lock.acquire(blocking=False):
            self.update_task(
                task_id,
                status="failed",
                error="Another task is already running. Please wait for it to complete.",
                message="Server busy -- another task is in progress.",
            )
            return

        self._running_task_id = task_id
        try:
            self._execute_single(
                task_id, config, input_path, output_dir, model_ckpt,
                keep_intermediates,
                report=lambda **kw: self.update_task(task_id, **kw),
            )
        finally:
            task = self._tasks.get(task_id)
            if task:
                task.finished_at = datetime.now().isoformat()
            self._running_task_id = None
            self._lock.release()

    def run_batch(self, batch_id: str) -> None:
        """
        Run all files of a batch sequentially under the single GPU lock.

        Failed files do not stop the batch -- remaining files continue.
        The batch is marked 'completed' if at least one file succeeded,
        'failed' only if all files failed or the lock was busy.
        """
        if not self._lock.acquire(blocking=False):
            batch = self._batches.get(batch_id)
            if batch:
                batch.status = "failed"
                batch.error = "Another task is already running. Please wait for it to complete."
                self._broadcast_batch_state(batch)
            return

        self._running_task_id = batch_id
        try:
            batch = self._batches.get(batch_id)
            if batch is None:
                return
            batch.started_at = datetime.now().isoformat()
            batch.status = "running"
            self._broadcast_batch_state(batch)

            for idx, entry in enumerate(batch.files):
                batch.current_index = idx
                self._running_task_id = entry.file_id

                def _make_report(_entry, _idx):
                    def _report(**kw):
                        self._batch_report(batch, _entry, _idx, kw)
                    return _report

                status = self._execute_single(
                    entry.file_id, entry.config, entry.input_path,
                    entry.output_dir, entry.model_ckpt,
                    entry.keep_intermediates, _make_report(entry, idx),
                )
                if status != "completed":
                    logger.warning(
                        "Batch %s: file %s failed -- continuing",
                        batch_id, entry.input_filename,
                    )

            # Determine final batch status
            failed_count = sum(1 for f in batch.files if f.status == "failed")
            batch.status = "completed" if failed_count < len(batch.files) else "failed"
            if failed_count == len(batch.files):
                batch.error = f"All {len(batch.files)} file(s) failed."
            batch.progress = 100.0
            batch.finished_at = datetime.now().isoformat()
            batch.message = f"Batch complete: {len(batch.files) - failed_count}/{len(batch.files)} succeeded"
            self._broadcast_batch_state(batch)
        except Exception as exc:
            logger.exception("Batch %s failed", batch_id)
            batch = self._batches.get(batch_id)
            if batch:
                batch.status = "failed"
                batch.error = str(exc)
                batch.finished_at = datetime.now().isoformat()
                self._broadcast_batch_state(batch)
        finally:
            self._running_task_id = None
            self._lock.release()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
task_manager = TaskManager()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Song Cover Pipeline",
    description="Web UI for song cover generation pipeline",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
static_dir = WEBUI_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    """
    Sanitize a user-provided filename for safe disk storage.

    Preserves Unicode (including Chinese), replaces path separators and
    other unsafe characters with underscores.  Strips leading/trailing
    whitespace and dots.  Ensures the result is non-empty.
    """
    # Replace truly unsafe characters (path separators, null bytes, etc.)
    unsafe = {'/', '\\', '\x00', ':', '*', '?', '"', '<', '>', '|'}
    safe = []
    for ch in name:
        if ch in unsafe:
            safe.append('_')
        else:
            safe.append(ch)
    result = ''.join(safe).strip().strip('.')
    if not result:
        result = "audio"
    # Truncate to reasonable length (keep extension intact later)
    if len(result) > 200:
        result = result[:200]
    return result


def _find_input_file(upload_dir: Path) -> Optional[Path]:
    """Find the first audio/video file in an upload directory."""
    if not upload_dir.is_dir():
        return None
    for entry in sorted(upload_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in ALLOWED_EXTS:
            return entry
    return None


def _available_output_files(task_id: str, output_files: Dict[str, str]) -> Dict[str, str]:
    """Return only output entries whose files still exist on disk."""
    output_root = (UPLOADS_DIR / task_id / "output").resolve()
    available: Dict[str, str] = {}
    for key, rel_path in (output_files or {}).items():
        try:
            relative = Path(rel_path)
            if relative.is_absolute():
                continue
            file_path = (output_root / relative).resolve()
            file_path.relative_to(output_root)
            if file_path.is_file():
                available[key] = str(relative)
        except (OSError, ValueError, TypeError):
            continue
    return available


def _resolve_project_path(p: str) -> Path:
    """Resolve a path-relative-to-project-root to an absolute Path."""
    pp = Path(p)
    return pp if pp.is_absolute() else _PROJECT_ROOT / pp


def _get_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes (fast, no recursion into symlinks)."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def _get_disk_usage() -> dict:
    """Return disk usage stats for the uploads directory."""
    try:
        stat = shutil.disk_usage(UPLOADS_DIR)
        uploads_size = _get_dir_size(UPLOADS_DIR)
        return {
            "uploads_size_gb": round(uploads_size / (1024**3), 2),
            "uploads_size_mb": round(uploads_size / (1024**2), 1),
            "disk_total_gb": round(stat.total / (1024**3), 1),
            "disk_free_gb": round(stat.free / (1024**3), 1),
            "disk_used_pct": round((1 - stat.free / stat.total) * 100, 1),
            "max_uploads_gb": MAX_TOTAL_UPLOADS_GB,
            "warning": uploads_size / (1024**3) > MAX_TOTAL_UPLOADS_GB,
        }
    except Exception:
        return {"error": "Cannot get disk usage"}


# =========================================================================
# Tiered cleanup functions
# =========================================================================
# Strategy:
#   1. Per-task: after success, immediately delete all stage dirs except
#      KEEP_STAGES_ON_SUCCESS (default: only 04_final_mix).
#   2. Periodic (on upload): delete intermediates of completed tasks older
#      than INTERMEDIATE_RETENTION_HOURS, and entire failed tasks older
#      than FAILED_RETENTION_HOURS.
#   3. Disk pressure: if free space < DISK_PRESSURE_FREE_GB, delete oldest
#      completed tasks first, then oldest failed tasks.
# =========================================================================

# All stage directories that the pipeline can produce
_ALL_STAGE_DIRS = [
    "00_extract_audio",
    "01_harmony_separation",
    "02_reverb_separation",
    "03_timbre_conversion",
    "04_final_mix",
]


def _cleanup_task_intermediates(task_dir: Path, keep_stages: List[str] = KEEP_STAGES_ON_SUCCESS) -> int:
    """
    Delete intermediate stage directories from a completed task,
    keeping only the stages listed in ``keep_stages``.

    Called immediately after a task completes successfully.

    Returns the number of stage directories removed.
    """
    if not task_dir.is_dir():
        return 0

    output_dir = task_dir / "output"
    if not output_dir.is_dir():
        return 0

    removed = 0
    for stage_name in _ALL_STAGE_DIRS:
        if stage_name in keep_stages:
            continue
        stage_path = output_dir / stage_name
        if stage_path.is_dir():
            size_before = _get_dir_size(stage_path)
            shutil.rmtree(stage_path, ignore_errors=True)
            if not stage_path.exists():
                removed += 1
                logger.info(
                    "Cleanup: deleted %s (freed %.1f MB)",
                    stage_path.relative_to(task_dir.parent),
                    size_before / (1024**2),
                )

    return removed


def _cleanup_periodic() -> dict:
    """
    Tiered periodic cleanup -- called before accepting new uploads.

    Deletion order (by urgency):
      1. Intermediates of completed tasks > INTERMEDIATE_RETENTION_HOURS
      2. Entire failed task dirs > FAILED_RETENTION_HOURS
      3. Entire completed task dirs > TASK_RETENTION_HOURS
      4. Disk pressure: oldest completed tasks (any age) until free space OK

    Returns a summary dict.
    """
    if not UPLOADS_DIR.is_dir():
        return {"removed_intermediates": 0, "removed_failed": 0,
                "removed_old": 0, "disk_pressure_deleted": 0}

    now = time.time()
    result = {"removed_intermediates": 0, "removed_failed": 0,
              "removed_old": 0, "disk_pressure_deleted": 0}

    # Collect task dirs with metadata
    tasks_meta: List[Dict] = []
    for task_dir in UPLOADS_DIR.iterdir():
        if not task_dir.is_dir():
            continue
        try:
            mtime = task_dir.stat().st_mtime
            # Determine status from task state (if still in memory) or from dir age
            task = task_manager.get_task(task_dir.name)
            status = task.status if task else "unknown"
            tasks_meta.append({
                "path": task_dir,
                "mtime": mtime,
                "age_h": round((now - mtime) / 3600, 1),
                "status": status,
            })
        except OSError:
            pass

    # ---- Phase 1: Delete intermediates of completed tasks (keep final only) ----
    for meta in tasks_meta:
        if meta["status"] != "completed":
            continue
        if meta["age_h"] < INTERMEDIATE_RETENTION_HOURS:
            continue
        output_dir = meta["path"] / "output"
        if not output_dir.is_dir():
            continue
        n = _cleanup_task_intermediates(meta["path"], KEEP_STAGES_ON_SUCCESS)
        result["removed_intermediates"] += n

    # ---- Phase 2: Delete entire failed task dirs ----
    for meta in tasks_meta:
        if meta["status"] not in ("failed", "unknown"):
            continue
        if meta["age_h"] < FAILED_RETENTION_HOURS:
            continue
        size = _get_dir_size(meta["path"])
        shutil.rmtree(meta["path"], ignore_errors=True)
        if not meta["path"].exists():
            result["removed_failed"] += 1
            logger.info("Cleanup: removed failed task %s (age: %.1f h, freed: %.1f MB)",
                        meta["path"].name, meta["age_h"], size / (1024**2))

    # ---- Phase 3: Delete old completed task dirs ----
    for meta in tasks_meta:
        if meta["status"] != "completed":
            continue
        if meta["age_h"] < TASK_RETENTION_HOURS:
            continue
        if not meta["path"].is_dir():
            continue
        size = _get_dir_size(meta["path"])
        shutil.rmtree(meta["path"], ignore_errors=True)
        if not meta["path"].exists():
            result["removed_old"] += 1
            logger.info("Cleanup: removed old task %s (age: %.1f h, freed: %.1f MB)",
                        meta["path"].name, meta["age_h"], size / (1024**2))

    # ---- Phase 4: Disk pressure -- aggressive if free space is low ----
    try:
        stat = shutil.disk_usage(UPLOADS_DIR)
        free_gb = stat.free / (1024**3)
    except Exception:
        free_gb = float("inf")

    if free_gb < DISK_PRESSURE_FREE_GB:
        # Delete oldest COMPLETED tasks first
        completed = sorted(
            [m for m in tasks_meta if m["status"] == "completed" and m["path"].is_dir()],
            key=lambda m: m["mtime"],
        )
        for meta in completed:
            if free_gb >= DISK_PRESSURE_FREE_GB * 1.5:
                break
            size = _get_dir_size(meta["path"])
            shutil.rmtree(meta["path"], ignore_errors=True)
            if not meta["path"].exists():
                result["disk_pressure_deleted"] += 1
                free_gb += size / (1024**3)
                logger.warning(
                    "Disk pressure: deleted completed task %s (freed %.1f MB, free now: %.1f GB)",
                    meta["path"].name, size / (1024**2), free_gb,
                )

        # If still tight, delete oldest failed tasks
        if free_gb < DISK_PRESSURE_FREE_GB:
            failed = sorted(
                [m for m in tasks_meta if m["status"] in ("failed", "unknown") and m["path"].is_dir()],
                key=lambda m: m["mtime"],
            )
            for meta in failed:
                if free_gb >= DISK_PRESSURE_FREE_GB:
                    break
                size = _get_dir_size(meta["path"])
                shutil.rmtree(meta["path"], ignore_errors=True)
                if not meta["path"].exists():
                    result["disk_pressure_deleted"] += 1
                    free_gb += size / (1024**3)
                    logger.warning(
                        "Disk pressure: deleted failed task %s (freed %.1f MB, free now: %.1f GB)",
                        meta["path"].name, size / (1024**2), free_gb,
                    )

    return result


# Backward-compatible alias
def _cleanup_old_tasks(max_age_hours: int = TASK_RETENTION_HOURS) -> int:
    """Legacy wrapper -- calls the periodic cleanup and returns total removed."""
    result = _cleanup_periodic()
    return result["removed_intermediates"] + result["removed_failed"] + \
           result["removed_old"] + result["disk_pressure_deleted"]


def _cleanup_uploads_on_startup() -> dict:
    """
    Remove ALL task directories and residual files from previous server runs.

    Called once at server startup.  Since all task state is in-memory,
    any on-disk artifacts from a prior run are orphaned and should be
    cleaned to free disk space and prevent stale data from being served.

    Returns a summary of what was removed.
    """
    if not UPLOADS_DIR.is_dir():
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        return {"removed_dirs": 0, "freed_mb": 0}

    removed = 0
    total_bytes = 0

    for entry in sorted(UPLOADS_DIR.iterdir()):
        try:
            if entry.is_dir():
                size = _get_dir_size(entry)
                shutil.rmtree(entry, ignore_errors=True)
                if not entry.exists():
                    total_bytes += size
                    removed += 1
                    logger.info("Startup cleanup: removed %s (%.1f MB)",
                                entry.name, size / (1024**2))
            elif entry.is_file():
                size = entry.stat().st_size
                entry.unlink(missing_ok=True)
                total_bytes += size
                logger.info("Startup cleanup: removed stray file %s", entry.name)
        except OSError:
            pass

    # Also clean any stale checkpoint files in the project output dir
    project_output = _PROJECT_ROOT / "output"
    if project_output.is_dir():
        for ckpt in project_output.rglob(".pipeline_checkpoint.json"):
            try:
                ckpt.unlink(missing_ok=True)
                logger.info("Startup cleanup: removed stale checkpoint %s", ckpt)
            except OSError:
                pass

    return {
        "removed_dirs": removed,
        "freed_mb": round(total_bytes / (1024**2), 1),
    }


def _scan_timbre_models() -> List[Dict[str, Any]]:
    """Discover available DDSP timbre model checkpoints."""
    models = []
    if not MODELS_DIR.is_dir():
        return models

    for pt_file in sorted(MODELS_DIR.rglob("*.pt")):
        # Skip pretrained/ directory models (f0 encoders, vocoders, etc.)
        if "pretrain" in pt_file.parts or "pretrained" in pt_file.parts:
            continue
        try:
            stat = pt_file.stat()
            models.append({
                "name": pt_file.stem,
                "path": str(pt_file.relative_to(_PROJECT_ROOT)),
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except OSError:
            pass

    return models


def _load_default_config() -> dict:
    """Load the default pipeline config from configs/ria_cover.yaml."""
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _generate_task_config(
    input_path: Path,
    output_dir: Path,
    model_ckpt: str,
    params: dict,
    steps: dict = None,
) -> dict:
    """
    Merge user-provided parameter overrides with the default config.

    The default config provides all the stable settings (model paths,
    MSST configs, etc.).  User params only override the exposed knobs.
    """
    if steps is None:
        steps = {}
    config = _load_default_config()

    # Update task section
    config["task"]["name"] = params.get("task_name", f"web-{uuid.uuid4().hex[:8]}")
    config["task"]["input_song"] = str(input_path)
    config["task"]["output_dir"] = str(output_dir)
    config["_project_root"] = str(_PROJECT_ROOT)

    # ---- Step selection: which pipeline stages to run ----
    # ``steps`` is sent as a top-level request field by both the single and
    # batch frontend flows.  Older clients may still put it in ``params``;
    # support both forms, preferring the explicit function argument.
    requested_steps = params.get("steps")
    if requested_steps is None:
        requested_steps = steps
    steps = requested_steps if isinstance(requested_steps, dict) else {}
    # Default: all enabled
    config.setdefault("harmony_separation", {})["enabled"] = steps.get("harmony", True)
    config.setdefault("reverb_separation", {})["enabled"]  = steps.get("reverb", True)
    config.setdefault("timbre_conversion", {})["enabled"]  = steps.get("timbre", True)
    # extract_audio and mixing are always enabled
    config.setdefault("extract_audio", {})["enabled"] = True
    config.setdefault("mixing", {})["enabled"] = True
    # linked_separation: only valid when BOTH harmony AND reverb are enabled
    linked_enabled = (
        config["harmony_separation"]["enabled"] and
        config["reverb_separation"]["enabled"] and
        config.get("linked_separation", {}).get("enabled", False)
    )
    config.setdefault("linked_separation", {})["enabled"] = linked_enabled

    # Timbre conversion overrides
    tc = config.setdefault("timbre_conversion", {})
    tc["model_ckpt"] = model_ckpt

    # ---- MSST uses torch.compile by default (matches project config) ----
    # DDSP is NOT compatible with torch.compile -- keep it disabled always.
    # DDSP use_amp also stays off (user reports incompatibility).
    config.setdefault("harmony_separation", {})["use_compile"] = True
    config.setdefault("reverb_separation", {})["use_compile"] = True
    config.setdefault("timbre_conversion", {})["use_compile"] = False
    config.setdefault("timbre_conversion", {})["use_amp"] = False

    param_map = {
        # User param key  ->  (section, config_key, coerce_type)
        "infer_step":            ("timbre_conversion", "infer_step", int),
        "t_start":               ("timbre_conversion", "t_start", float),
        "method":                ("timbre_conversion", "method", str),
        "pitch_extractor":       ("timbre_conversion", "pitch_extractor", str),
        "key":                   ("timbre_conversion", "key", float),
        "formant_shift":         ("timbre_conversion", "formant_shift", float),
        "vocal_register_shift":  ("timbre_conversion", "vocal_register_shift", float),
        "f0_min":                ("timbre_conversion", "f0_min", float),
        "f0_max":                ("timbre_conversion", "f0_max", float),
        "threshold":             ("timbre_conversion", "threshold", float),
        "spk_id":                ("timbre_conversion", "spk_id", int),
        "segment_batch_size":    ("timbre_conversion", "segment_batch_size", int),
        "vocal_gain":            ("mixing", "vocal_gain", float),
        "instrumental_gain":     ("mixing", "instrumental_gain", float),
        "reverb_gain":           ("mixing", "reverb_gain", float),
        "normalize_output":      ("mixing", "normalize_output", bool),
        "output_format":         ("mixing", "output_format", str),
        "chunk_batch_harmony":   ("harmony_separation", "chunk_batch", int),
        "chunk_batch_reverb":    ("reverb_separation", "chunk_batch", int),
    }

    for user_key, (section, config_key, coerce) in param_map.items():
        if user_key in params and params[user_key] is not None:
            try:
                config.setdefault(section, {})[config_key] = coerce(params[user_key])
            except (ValueError, TypeError):
                pass  # ignore bad values, keep default

    # ---- use_compile checkbox controls MSST only (DDSP is not compatible) ----
    if "use_compile" in params and params["use_compile"] is not None:
        val = bool(params["use_compile"])
        config.setdefault("harmony_separation", {})["use_compile"] = val
        config.setdefault("reverb_separation", {})["use_compile"] = val
        # NEVER enable on timbre_conversion -- DDSP is incompatible

    return config


# ---------------------------------------------------------------------------
# API: Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Server and GPU health check."""
    gpu_info = {"available": False, "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = torch.cuda.get_device_name(0)
            gpu_info["memory_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
            gpu_info["memory_used_gb"] = round(
                torch.cuda.memory_allocated(0) / (1024**3), 2
            )
    except ImportError:
        pass

    return {
        "status": "ok",
        "gpu": gpu_info,
        "task_running": task_manager.is_running(),
        "models_available": len(_scan_timbre_models()),
        "disk": _get_disk_usage(),
    }


# ---------------------------------------------------------------------------
# API: Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    """List available timbre models found in models/DDSP/."""
    models = _scan_timbre_models()
    return {
        "models": models,
        "default": "models/DDSP/paipai.pt" if any(
            m["name"] == "paipai" for m in models
        ) else (models[0]["path"] if models else ""),
    }


# ---------------------------------------------------------------------------
# API: Default config values (exposed parameters)
# ---------------------------------------------------------------------------

@app.get("/api/config/defaults")
async def get_defaults():
    """Return the default values for all user-exposed parameters."""
    config = _load_default_config()
    tc = config.get("timbre_conversion", {})
    mix = config.get("mixing", {})
    hs = config.get("harmony_separation", {})
    rs = config.get("reverb_separation", {})

    return {
        "timbre_conversion": {
            "infer_step": tc.get("infer_step", 100),
            "t_start": tc.get("t_start", 0.4),
            "method": tc.get("method", "rk4"),
            "pitch_extractor": tc.get("pitch_extractor", "rmvpe"),
            "key": tc.get("key", 0),
            "formant_shift": tc.get("formant_shift", 0),
            "vocal_register_shift": tc.get("vocal_register_shift", 0),
            "f0_min": tc.get("f0_min", 65),
            "f0_max": tc.get("f0_max", 800),
            "threshold": tc.get("threshold", -60),
            "spk_id": tc.get("spk_id", 1),
            "use_compile": tc.get("use_compile", False),
            "use_amp": tc.get("use_amp", False),
            "segment_batch_size": tc.get("segment_batch_size", 1),
        },
        "mixing": {
            "vocal_gain": mix.get("vocal_gain", 0.0),
            "instrumental_gain": mix.get("instrumental_gain", 0.0),
            "reverb_gain": mix.get("reverb_gain", -3.0),
            "normalize_output": mix.get("normalize_output", True),
            "output_format": mix.get("output_format", "wav"),
        },
        "separation": {
            "chunk_batch_harmony": hs.get("chunk_batch", 16),
            "chunk_batch_reverb": rs.get("chunk_batch", 8),
        },
        "pitch_extractors": ["rmvpe", "fcpe", "parselmouth", "dio", "harvest", "crepe"],
        "methods": ["euler", "rk4"],
    }


# ---------------------------------------------------------------------------
# API: File upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload an input song (audio or video file).

    The file is streamed to disk in chunks -- it never resides fully in
    server memory.  A size limit is enforced to prevent disk exhaustion.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    # --- Auto-cleanup stale tasks before accepting new upload ---
    _cleanup_old_tasks()

    # Check disk space before accepting the upload
    try:
        disk = _get_disk_usage()
        if disk.get("warning"):
            logger.warning(
                "Uploads directory exceeds %d GB limit -- consider cleaning up.",
                MAX_TOTAL_UPLOADS_GB,
            )
    except Exception:
        pass

    # Create a staging task ID for the upload
    upload_id = uuid.uuid4().hex[:12]
    task_upload_dir = UPLOADS_DIR / upload_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = f"{_sanitize_filename(Path(file.filename).stem)}{ext}"
    file_path = task_upload_dir / safe_filename
    total_bytes = 0

    try:
        # ---- Stream to disk in chunks (never hold full file in RAM) ----
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    f.close()
                    shutil.rmtree(task_upload_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {MAX_UPLOAD_SIZE_BYTES // (1024**3)} GB",
                    )
                f.write(chunk)

    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(task_upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    file_size_mb = round(total_bytes / (1024 * 1024), 1)

    logger.info("Uploaded: %s (%.1f MB) -> %s", file.filename, file_size_mb, upload_id)

    return {
        "upload_id": upload_id,
        "original_name": file.filename,
        "filename": safe_filename,
        "size_mb": file_size_mb,
        "is_video": ext in VIDEO_EXTS,
    }


# ---------------------------------------------------------------------------
# API: Create & run task
# ---------------------------------------------------------------------------

@app.post("/api/tasks")
async def create_task(request: Request):
    """Create a new pipeline task and start processing."""
    if task_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A task is already running. Please wait for it to complete.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    upload_id = body.get("upload_id", "")
    params = body.get("params", {})
    keep_intermediates = body.get("keep_intermediates", False)
    steps = body.get("steps", {})

    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")

    upload_dir = UPLOADS_DIR / upload_id
    if not upload_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload not found")

    # Find the uploaded file
    input_path = _find_input_file(upload_dir)
    if input_path is None:
        raise HTTPException(status_code=404, detail="No input file found for this upload")

    # Determine model checkpoint
    model_ckpt = params.get("model_ckpt", "models/DDSP/paipai.pt")
    model_path = _resolve_project_path(model_ckpt)
    if not model_path.is_file():
        default_model = _PROJECT_ROOT / "models" / "DDSP" / "paipai.pt"
        if default_model.is_file():
            model_ckpt = "models/DDSP/paipai.pt"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model checkpoint not found: {model_ckpt}",
            )

    model_name = Path(model_ckpt).stem

    # Create task
    task_id = upload_id  # reuse upload_id as task_id for simplicity

    # Prevent re-submitting the same upload if a task already exists
    existing = task_manager.get_task(task_id)
    if existing is not None and existing.status == "running":
        raise HTTPException(
            status_code=409,
            detail="This upload is currently being processed. Please wait for it to complete.",
        )

    task = task_manager.create_task(
        input_filename=input_path.name,
        model_name=model_name,
    )
    # Override the auto-generated ID to match upload_id
    old_id = task.id
    task.id = task_id
    task_manager._tasks[task_id] = task
    task_manager._ws_subscribers[task_id] = task_manager._ws_subscribers.pop(old_id, [])
    if old_id in task_manager._tasks:
        del task_manager._tasks[old_id]

    # Set up output directory
    output_dir = upload_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate config
    config = _generate_task_config(
        input_path=input_path,
        output_dir=output_dir,
        model_ckpt=model_ckpt,
        params=params,
        steps=steps,
    )

    # Save a copy of the generated config for reference
    config_path = upload_dir / "generated_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)

    logger.info("Task %s created -- model: %s, input: %s", task_id, model_name, input_path.name)

    # If somehow a task is already running (race), catch it in the thread
    if task_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A task started between checks. Please try again.",
        )

    # Start pipeline in background thread
    thread = threading.Thread(
        target=task_manager.run_task,
        args=(task_id, config, input_path, output_dir, model_ckpt, keep_intermediates),
        daemon=True,
        name=f"pipeline-{task_id}",
    )
    thread.start()

    return {
        "task_id": task_id,
        "status": "pending",
        "ws_url": f"/ws/tasks/{task_id}",
    }


# ---------------------------------------------------------------------------
# API: Cancel / reset a stuck task
# ---------------------------------------------------------------------------

@app.post("/api/tasks/{task_id}/reset")
async def reset_task(task_id: str):
    """Force-reset a task that appears stuck (admin recovery)."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # Only allow resetting non-running or failed tasks
    if task.status == "running" and task_manager.is_running():
        raise HTTPException(
            status_code=400,
            detail="Cannot reset a running task. Wait for it to complete.",
        )

    task.status = "failed"
    task.error = "Reset by user"
    task.message = "Task was manually reset."
    task.finished_at = datetime.now().isoformat()
    task_manager.update_task(task_id)

    return {"status": "ok", "task_id": task_id, "new_status": "failed"}


# ---------------------------------------------------------------------------
# API: Task status
# ---------------------------------------------------------------------------

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get the current state of a task."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.id,
        "status": task.status,
        "progress": task.progress,
        "stage": task.stage,
        "message": task.message,
        "output_files": _available_output_files(task.id, task.output_files),
        "error": task.error,
        "input_filename": task.input_filename,
        "model_name": task.model_name,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


# ---------------------------------------------------------------------------
# API: Task log
# ---------------------------------------------------------------------------

@app.get("/api/tasks/{task_id}/log")
async def get_task_log(task_id: str):
    """Return the detailed execution log for a task."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    log_path = UPLOADS_DIR / task_id / "task.log"
    result_path = UPLOADS_DIR / task_id / "task_result.json"

    log_text = ""
    if log_path.is_file():
        try:
            log_text = log_path.read_text(encoding="utf-8")
        except Exception:
            log_text = "(log file is binary or unreadable)"

    result_data = None
    if result_path.is_file():
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "task_id": task_id,
        "status": task.status,
        "log": log_text,
        "log_lines": log_text.count("\n") + 1 if log_text else 0,
        "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
        "result": result_data,
    }


# ---------------------------------------------------------------------------
# API: Download output file
# ---------------------------------------------------------------------------

@app.get("/api/tasks/{task_id}/output/{filename:path}")
async def download_output(task_id: str, filename: str):
    """
    Download an output file from a completed task.

    Uses chunked streaming -- the file is read from disk in small blocks
    so it never fully resides in server memory.  Supports HTTP Range
    requests for resumable downloads.
    """
    from fastapi.responses import StreamingResponse

    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")

    # Depending on the ASGI server/proxy, percent-encoded path components may
    # reach the endpoint either decoded or still encoded.
    filename = unquote(filename)

    # Resolve only paths inside this task's output directory.  The frontend
    # sends a relative stage path (for example, 04_final_mix/cover.wav).
    upload_dir = UPLOADS_DIR / task_id
    output_dir = upload_dir / "output"

    requested_path = Path(filename)
    if requested_path.is_absolute() or ".." in requested_path.parts:
        raise HTTPException(status_code=400, detail="Invalid output file path")

    try:
        output_root = output_dir.resolve()
        file_path = (output_root / requested_path).resolve()
        file_path.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid output file path")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    file_size = file_path.stat().st_size
    media_type = (
        "audio/wav" if file_path.suffix == ".wav" else
        "audio/flac" if file_path.suffix == ".flac" else
        "application/octet-stream"
    )

    def _file_iterator(path: Path, chunk_size: int = 1024 * 1024):
        """Yield file content in chunks -- never loads full file into RAM."""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # HTTP headers are Latin-1, so a raw Chinese filename raises a server
    # error.  Keep an ASCII fallback and provide the real UTF-8 name via the
    # RFC 6266 filename* parameter.
    ascii_name = "".join(
        ch if 32 <= ord(ch) < 127 and ch not in {'"', "\\"} else "_"
        for ch in file_path.name
    ).strip() or "download"
    content_disposition = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(file_path.name, safe='')}"
    )

    return StreamingResponse(
        _file_iterator(file_path),
        media_type=media_type,
        headers={
            "Content-Disposition": content_disposition,
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# API: Preview / stream audio (chunked, seekable)
# ---------------------------------------------------------------------------

@app.get("/api/tasks/{task_id}/preview")
async def preview_audio(task_id: str, request: Request):
    """
    Stream the final cover audio for in-browser playback.

    Supports HTTP Range requests so the browser can seek without
    downloading the entire file first.
    """
    from fastapi.responses import StreamingResponse

    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task not yet completed")

    # Find the cover file
    upload_dir = UPLOADS_DIR / task_id
    output_dir = upload_dir / "output"
    mix_dir = output_dir / "04_final_mix"

    cover_files = list(mix_dir.glob("*_cover.wav")) if mix_dir.is_dir() else []
    if not cover_files:
        cover_files = list(output_dir.rglob("*_cover.wav"))
    if not cover_files:
        if mix_dir.is_dir():
            cover_files = list(mix_dir.glob("*.wav"))
    if not cover_files:
        raise HTTPException(status_code=404, detail="Cover file not found")

    cover_path = cover_files[0]
    file_size = cover_path.stat().st_size

    # ---- Handle HTTP Range requests for seekable playback ----
    range_header = request.headers.get("range", "")
    start = 0
    end = file_size - 1

    if range_header:
        try:
            if not range_header.startswith("bytes=") or "," in range_header:
                raise ValueError
            range_str = range_header[6:]
            start_text, end_text = range_str.split("-", 1)
            if not start_text and not end_text:
                raise ValueError
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else file_size - 1
            else:
                # Suffix range: bytes=-N means the final N bytes.
                suffix_length = int(end_text)
                if suffix_length <= 0:
                    raise ValueError
                start = max(file_size - suffix_length, 0)
                end = file_size - 1
            if start < 0 or start >= file_size:
                raise ValueError
            end = min(end, file_size - 1)
            if end < start:
                raise ValueError
        except (ValueError, IndexError):
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )

    chunk_size = end - start + 1
    status_code = 206 if range_header else 200

    def _range_iterator(path: Path, offset: int, length: int, buf_size: int = 256 * 1024):
        """Yield a byte range from a file in chunks."""
        with open(path, "rb") as f:
            f.seek(offset)
            remaining = length
            while remaining > 0:
                n = min(buf_size, remaining)
                data = f.read(n)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Type": "audio/wav",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Cache-Control": "no-cache",
    }
    if range_header:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

    return StreamingResponse(
        _range_iterator(cover_path, start, chunk_size),
        status_code=status_code,
        media_type="audio/wav",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# API: Cleanup old tasks (manual trigger)
# ---------------------------------------------------------------------------

@app.post("/api/cleanup")
async def cleanup_tasks():
    """Manually trigger tiered cleanup of old task files."""
    result = _cleanup_periodic()
    disk = _get_disk_usage()
    return {
        "cleanup": result,
        "disk": disk,
    }


# ---------------------------------------------------------------------------
# API: Batch processing
# ---------------------------------------------------------------------------

@app.post("/api/batch")
async def create_batch(request: Request):
    """Create a batch job with multiple upload IDs and start processing."""
    if task_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A task is already running. Please wait for it to complete.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    upload_ids = body.get("upload_ids", [])
    params = body.get("params", {})
    keep_intermediates = body.get("keep_intermediates", False)
    steps = body.get("steps", {})

    if not upload_ids or not isinstance(upload_ids, list):
        raise HTTPException(status_code=400, detail="upload_ids (non-empty list) is required")
    if len(upload_ids) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_FILES} files per batch. Got {len(upload_ids)}.",
        )

    # Determine model checkpoint (shared across all files)
    model_ckpt = params.get("model_ckpt", "models/DDSP/paipai.pt")
    model_path = _resolve_project_path(model_ckpt)
    if not model_path.is_file():
        default_model = _PROJECT_ROOT / "models" / "DDSP" / "paipai.pt"
        if default_model.is_file():
            model_ckpt = "models/DDSP/paipai.pt"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Model checkpoint not found: {model_ckpt}",
            )

    # Validate and prepare each file
    files_meta = []
    seen_ids = set()
    for uid in upload_ids:
        if not isinstance(uid, str) or not uid.strip():
            raise HTTPException(status_code=400, detail=f"Invalid upload_id: {uid}")
        uid = uid.strip()
        if uid in seen_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate upload_id: {uid}")
        seen_ids.add(uid)

        upload_dir = UPLOADS_DIR / uid
        if not upload_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"Upload not found: {uid}")

        input_path = _find_input_file(upload_dir)
        if input_path is None:
            raise HTTPException(status_code=404, detail=f"No input file found for upload: {uid}")

        # Prevent re-submitting already-running/completed tasks
        existing = task_manager.get_task(uid)
        if existing is not None and existing.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Upload {uid} is currently being processed. Please wait for it to complete.",
            )

        output_dir = upload_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        config = _generate_task_config(
            input_path=input_path,
            output_dir=output_dir,
            model_ckpt=model_ckpt,
            params=params,
            steps=steps,
        )

        # Save generated config per upload dir
        config_path = upload_dir / "generated_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False)

        files_meta.append({
            "upload_id": uid,
            "input_filename": input_path.name,
            "input_path": input_path,
            "output_dir": output_dir,
            "config": config,
            "model_ckpt": model_ckpt,
            "keep_intermediates": keep_intermediates,
        })

    # Create batch
    batch = task_manager.create_batch(files_meta)
    logger.info("Batch %s created with %d file(s)", batch.id, len(batch.files))

    # Double-check lock (race guard)
    if task_manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A task started between checks. Please try again.",
        )

    # Start batch in background thread
    thread = threading.Thread(
        target=task_manager.run_batch,
        args=(batch.id,),
        daemon=True,
        name=f"batch-{batch.id}",
    )
    thread.start()

    return {
        "batch_id": batch.id,
        "status": "pending",
        "ws_url": f"/ws/batches/{batch.id}",
        "files": [
            {"file_id": f.file_id, "input_filename": f.input_filename, "status": f.status}
            for f in batch.files
        ],
    }


@app.get("/api/batches/{batch_id}")
async def get_batch(batch_id: str):
    """Get the current state of a batch job (REST polling fallback)."""
    batch = task_manager.get_batch(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    payload = task_manager._batch_payload(batch)
    del payload["type"]  # REST response doesn't need the WS type field
    return payload


@app.websocket("/ws/batches/{batch_id}")
async def batch_websocket_progress(ws: WebSocket, batch_id: str):
    """WebSocket endpoint for real-time batch progress updates."""
    await ws.accept()

    batch = task_manager.get_batch(batch_id)
    if batch is None:
        await ws.send_json({"type": "error", "message": "Batch not found"})
        await ws.close()
        return

    task_manager.subscribe_batch(batch_id, ws)

    try:
        # Send current state immediately
        await ws.send_json(task_manager._batch_payload(batch))

        # Keep connection alive, wait for messages
        while True:
            try:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text("pong")
                elif data == "get_status":
                    batch = task_manager.get_batch(batch_id)
                    if batch:
                        await ws.send_json(task_manager._batch_payload(batch))
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        task_manager.unsubscribe_batch(batch_id, ws)


# ---------------------------------------------------------------------------
# WebSocket: Progress stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/tasks/{task_id}")
async def websocket_progress(ws: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates."""
    await ws.accept()

    task = task_manager.get_task(task_id)
    if task is None:
        await ws.send_json({"type": "error", "message": "Task not found"})
        await ws.close()
        return

    task_manager.subscribe(task_id, ws)

    try:
        # Send current state immediately
        await ws.send_json({
            "type": "progress",
            "task_id": task.id,
            "status": task.status,
            "progress": task.progress,
            "stage": task.stage,
            "message": task.message,
            "output_files": _available_output_files(task.id, task.output_files) or None,
            "error": task.error,
        })

        # Keep connection alive, wait for messages (client may send pings)
        while True:
            try:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text("pong")
                elif data == "get_status":
                    task = task_manager.get_task(task_id)
                    if task:
                        await ws.send_json({
                            "type": "progress",
                            "task_id": task.id,
                            "status": task.status,
                            "progress": task.progress,
                            "stage": task.stage,
                            "message": task.message,
                            "output_files": _available_output_files(task.id, task.output_files) or None,
                            "error": task.error,
                        })
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        task_manager.unsubscribe(task_id, ws)


# ---------------------------------------------------------------------------
# Root: Serve SPA
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_spa():
    """Serve the single-page app."""
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Web UI not found</h1><p>static/index.html is missing.</p>", status_code=404)
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))

    # ---- Startup: clean all artifacts from previous runs ----
    cleanup_result = _cleanup_uploads_on_startup()

    print("=" * 60)
    print("  Song Cover Pipeline -- Web UI Server")
    print("=" * 60)
    print(f"  Project root:   {_PROJECT_ROOT}")
    print(f"  Listening on:   http://{host}:{port}")
    print(f"  Models found:   {len(_scan_timbre_models())}")
    print(f"  Startup cleanup: {cleanup_result['removed_dirs']} dir(s) removed, "
          f"{cleanup_result['freed_mb']} MB freed")
    print("=" * 60)

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
