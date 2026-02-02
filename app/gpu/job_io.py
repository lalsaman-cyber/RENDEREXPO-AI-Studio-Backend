from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


OUTPUTS_ROOT_DEFAULT = "/workspace-data/outputs"


@dataclass(frozen=True)
class JobRef:
    date: str
    job_id: str

    @property
    def rel_dir(self) -> str:
        return f"{self.date}/{self.job_id}"


def utc_datestr() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def outputs_root() -> Path:
    # Absolute, persistent, POD-safe root (locked by user)
    return Path(os.getenv("RENDEREXPO_OUTPUTS_ROOT", OUTPUTS_ROOT_DEFAULT)).resolve()


def job_dir(job: JobRef) -> Path:
    return outputs_root() / job.date / job.job_id


def ensure_job_dir(job: JobRef) -> Path:
    d = job_dir(job)
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta_path(job: JobRef) -> Path:
    return job_dir(job) / "meta.json"


def artifact_path(job: JobRef, filename: str) -> Path:
    return job_dir(job) / filename


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic replace: write to temp file in same dir then rename
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False, suffix=".tmp") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def init_meta(job: JobRef, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create meta.json if missing, without destroying existing fields.
    """
    d = ensure_job_dir(job)
    _ = d  # keep
    meta = read_json(meta_path(job))
    if meta.get("job_id") != job.job_id:
        meta["job_id"] = job.job_id
    if meta.get("date") != job.date:
        meta["date"] = job.date

    # keep a minimal request echo for debugging (safe + useful)
    meta.setdefault("request", {})
    for k in ("task", "profile", "seed", "width", "height"):
        if k in payload:
            meta["request"][k] = payload[k]

    meta.setdefault("status", "queued")
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    atomic_write_json(meta_path(job), meta)
    return meta


def update_meta(job: JobRef, **updates: Any) -> Dict[str, Any]:
    meta = read_json(meta_path(job))
    meta.update(updates)
    meta["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(meta_path(job), meta)
    return meta


def set_status(job: JobRef, status: str, *, message: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": status}
    if message is not None:
        payload["message"] = message
    return update_meta(job, **payload)
