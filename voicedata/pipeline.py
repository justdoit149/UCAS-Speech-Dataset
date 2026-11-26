"""流水线编排、锁与日志。"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path

import psutil

from .adapters import build_adapters
from .asr import run_asr
from .config import STAGES, Config
from .db import Database
from .decode import run_decode
from .export import run_export
from .ingest import register
from .models import ensure_models
from .noise import run_noise
from .sv import run_sv
from .vad import run_vad

logger = logging.getLogger(__name__)


class PipelineBusy(RuntimeError):
    pass


def setup_logging(work_dir: Path, level: int = logging.INFO) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    log_path = str((work_dir / "pipeline.log").resolve())
    if not any(
        isinstance(h, logging.FileHandler)
        and os.path.abspath(h.baseFilename) == os.path.abspath(log_path)
        for h in root.handlers
    ):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def acquire_lock(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = work_dir / ".lock"
    info = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    if lock.exists():
        try:
            old = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            old = {}
        pid = int(old.get("pid", -1))
        if pid > 0 and old.get("host") == info["host"] and psutil.pid_exists(pid):
            raise PipelineBusy(f"已有进程在跑同一个 job（pid={pid}）：{work_dir}")
        logger.warning("接管过期锁文件：%s", lock)
    tmp = lock.with_name(".lock.tmp")
    tmp.write_text(json.dumps(info), encoding="utf-8")
    os.replace(tmp, lock)
    return lock


def release_lock(lock: Path) -> None:
    try:
        if lock.exists():
            lock.unlink()
    except OSError:
        pass


def file_ids_for(db: Database, pattern: str | None = None) -> list[str]:
    rows = db.files()
    if pattern:
        rows = [r for r in rows if pattern.lower() in r["audio_file"].lower()]
    return [r["file_id"] for r in rows]


class Pipeline:
    def __init__(self, cfg: Config, job: str):
        self.cfg = cfg
        self.job = job
        self.work = cfg.job_dir("work", job)
        setup_logging(self.work)
        self.lock = acquire_lock(self.work)
        self.db = Database(self.work / "state.db")
        self.adapters = build_adapters(cfg)

    def close(self) -> None:
        try:
            self.db.close()
        finally:
            release_lock(self.lock)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _retry_failed(self, file_ids: list[str], stages: list[str]) -> None:
        for fid in file_ids:
            for stage in stages:
                row = self.db.get_stage(fid, stage)
                if row and row["status"] in ("failed", "partial"):
                    self.db.execute(
                        "UPDATE file_stages SET status='pending', error=NULL "
                        "WHERE file_id=? AND stage=?",
                        (fid, stage),
                    )
                    if stage in ("sv", "asr", "noise"):
                        self.db.execute(
                            f"UPDATE segments SET {stage}_state='pending' "
                            "WHERE file_id=? AND "
                            f"(COALESCE({stage}_state,'pending') IN ('failed','running','partial'))",
                            (fid,),
                        )
                    if stage == "decode":
                        self.db.execute(
                            "UPDATE files SET decode_state='pending' WHERE file_id=?", (fid,)
                        )
        self.db.commit()

    def run(
        self,
        pattern: str | None = None,
        start_s: float | None = None,
        duration_s: float | None = None,
        stages: list[str] | None = None,
        force: bool = False,
        retry_failed: bool = False,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[str]:
        stages = stages or list(STAGES)
        bad = [s for s in stages if s not in STAGES]
        if bad:
            raise ValueError(f"未知阶段：{bad}")
        ordered = [s for s in STAGES if s in stages]

        # 先检查/下载模型，避免已经开始登记文件或解码后才发现模型缺失
        ensure_models(self.cfg, ordered)

        file_ids = register(self.db, self.cfg, pattern, start_s, duration_s)
        if limit:
            file_ids = file_ids[: int(limit)]
        if retry_failed:
            self._retry_failed(file_ids, ordered)

        for stage in ordered:
            if stage == "decode":
                run_decode(self.db, self.cfg, self.job, file_ids, force)
            elif stage == "vad":
                run_vad(self.db, self.cfg, self.job, file_ids, self.adapters, force)
            elif stage == "sv":
                run_sv(self.db, self.cfg, self.job, file_ids, self.adapters, force)
            elif stage == "asr":
                run_asr(self.db, self.cfg, self.job, file_ids, self.adapters, force)
            elif stage == "noise":
                run_noise(self.db, self.cfg, self.job, file_ids, self.adapters, force)
            elif stage == "export":
                run_export(self.db, self.cfg, self.job, file_ids, threshold, force)
        return file_ids
