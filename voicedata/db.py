"""SQLite 状态库：文件、阶段、片段与导出记录。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS files(
    file_id TEXT PRIMARY KEY,
    source_key TEXT UNIQUE,
    rel_path TEXT,
    audio_file TEXT,
    stem TEXT,
    size_bytes INTEGER,
    mtime_ns INTEGER,
    source_duration_s REAL,
    offset_s REAL DEFAULT 0,
    excerpt_duration_s REAL,
    decoded_duration_s REAL,
    decoded_wav TEXT,
    decode_state TEXT DEFAULT 'pending',
    decode_params_hash TEXT,
    error TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS file_stages(
    file_id TEXT,
    stage TEXT,
    params_hash TEXT,
    status TEXT,
    attempts INTEGER DEFAULT 0,
    error TEXT,
    stats_json TEXT,
    updated_at TEXT,
    PRIMARY KEY(file_id, stage)
);
CREATE TABLE IF NOT EXISTS segments(
    segment_id TEXT PRIMARY KEY,
    file_id TEXT,
    vad_hash TEXT,
    seq INTEGER,
    start_s REAL,
    end_s REAL,
    duration_s REAL,
    wav_path TEXT,
    sv_state TEXT DEFAULT 'pending',
    sv_params_hash TEXT,
    sv_emb BLOB,
    sv_score REAL,
    sv_tidx INTEGER,
    asr_state TEXT DEFAULT 'pending',
    asr_params_hash TEXT,
    asr_text TEXT,
    asr_raw TEXT,
    noise_state TEXT DEFAULT 'pending',
    noise_params_hash TEXT,
    noisy_prob REAL,
    noisy_flag INTEGER,
    error TEXT,
    updated_at TEXT,
    UNIQUE(file_id, vad_hash, seq)
);
CREATE INDEX IF NOT EXISTS idx_segments_file ON segments(file_id, vad_hash, seq);
CREATE INDEX IF NOT EXISTS idx_segments_sv ON segments(sv_state, sv_score);
CREATE INDEX IF NOT EXISTS idx_segments_asr ON segments(asr_state);
CREATE INDEX IF NOT EXISTS idx_segments_noise ON segments(noise_state);
CREATE TABLE IF NOT EXISTS exports(
    file_id TEXT PRIMARY KEY,
    status TEXT,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    threshold REAL,
    export_params_hash TEXT,
    segment_count INTEGER,
    skipped_count INTEGER,
    generated_at TEXT,
    error TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Database:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        if readonly:
            # 只读打开：不建表、不迁移、不把 running 改回 pending，避免干扰正在跑的任务
            if not self.path.exists():
                raise FileNotFoundError(f"状态库不存在：{self.path}")
            self.conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=5)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA busy_timeout=5000")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.recover_stale()
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.query("PRAGMA table_info(files)")}
        if "excerpt_duration_s" not in cols:
            self.execute("ALTER TABLE files ADD COLUMN excerpt_duration_s REAL")

        # 旧版 exports 记录的是 JSON 路径/哈希；新版统一叫 artifact（zip 或兼容 JSON）
        export_cols = {r["name"] for r in self.query("PRAGMA table_info(exports)")}
        if "artifact_path" not in export_cols:
            self.execute("ALTER TABLE exports ADD COLUMN artifact_path TEXT")
        if "artifact_sha256" not in export_cols:
            self.execute("ALTER TABLE exports ADD COLUMN artifact_sha256 TEXT")
        if "json_path" in export_cols:
            self.execute(
                "UPDATE exports SET artifact_path=json_path "
                "WHERE artifact_path IS NULL AND json_path IS NOT NULL"
            )
        if "json_sha256" in export_cols:
            self.execute(
                "UPDATE exports SET artifact_sha256=json_sha256 "
                "WHERE artifact_sha256 IS NULL AND json_sha256 IS NOT NULL"
            )
        self.conn.commit()

    # ---------- 基础 ----------
    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        self.conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, tuple(params)))

    def one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchone()

    # ---------- meta ----------
    def set_meta(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    # ---------- 崩溃恢复 ----------
    def recover_stale(self) -> None:
        self.execute("UPDATE file_stages SET status='pending' WHERE status='running'")
        for col in ("sv_state", "asr_state", "noise_state"):
            self.execute(f"UPDATE segments SET {col}='pending' WHERE {col}='running'")
        self.execute("UPDATE files SET decode_state='pending' WHERE decode_state='running'")

    # ---------- files ----------
    def upsert_file(
        self,
        file_id: str,
        source_key: str,
        rel_path: str,
        audio_file: str,
        stem: str,
        size_bytes: int,
        mtime_ns: int,
        source_duration_s: float,
        offset_s: float,
        excerpt_duration_s: float | None = None,
    ) -> bool:
        """返回 True 表示新建，False 表示已存在。文件变化时重置解码状态。"""
        old = self.one("SELECT * FROM files WHERE source_key=?", (source_key,))
        if old is None:
            self.execute(
                "INSERT INTO files(file_id, source_key, rel_path, audio_file, stem, "
                "size_bytes, mtime_ns, source_duration_s, offset_s, excerpt_duration_s, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (file_id, source_key, rel_path, audio_file, stem, size_bytes,
                 mtime_ns, source_duration_s, offset_s, excerpt_duration_s, utcnow()),
            )
            self.conn.commit()
            return True
        changed = old["size_bytes"] != size_bytes or old["mtime_ns"] != mtime_ns
        self.execute(
            "UPDATE files SET rel_path=?, audio_file=?, stem=?, size_bytes=?, mtime_ns=?, "
            "source_duration_s=?, offset_s=?, excerpt_duration_s=?, updated_at=? WHERE source_key=?",
            (rel_path, audio_file, stem, size_bytes, mtime_ns, source_duration_s,
             offset_s, excerpt_duration_s, utcnow(), source_key),
        )
        if changed:
            self.execute(
                "UPDATE files SET decode_state='pending', decoded_wav=NULL, error=NULL "
                "WHERE source_key=?",
                (source_key,),
            )
        self.conn.commit()
        return False

    def get_file(self, file_id: str) -> Optional[sqlite3.Row]:
        return self.one("SELECT * FROM files WHERE file_id=?", (file_id,))

    def get_file_by_key(self, source_key: str) -> Optional[sqlite3.Row]:
        return self.one("SELECT * FROM files WHERE source_key=?", (source_key,))

    def files(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM files ORDER BY rel_path, offset_s")

    def file_ids(self) -> list[str]:
        return [r["file_id"] for r in self.files()]

    def set_decode(
        self,
        file_id: str,
        state: str,
        params_hash: str,
        decoded_wav: Optional[str] = None,
        decoded_duration_s: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self.execute(
            "UPDATE files SET decode_state=?, decode_params_hash=?, decoded_wav=?, "
            "decoded_duration_s=?, error=?, updated_at=? WHERE file_id=?",
            (state, params_hash, decoded_wav, decoded_duration_s, error, utcnow(), file_id),
        )
        self.conn.commit()

    # ---------- file_stages ----------
    def get_stage(self, file_id: str, stage: str) -> Optional[sqlite3.Row]:
        return self.one(
            "SELECT * FROM file_stages WHERE file_id=? AND stage=?", (file_id, stage)
        )

    def begin_stage(self, file_id: str, stage: str, params_hash: str) -> None:
        self.execute(
            "INSERT INTO file_stages(file_id, stage, params_hash, status, attempts, updated_at) "
            "VALUES(?,?,?,'running',1,?) "
            "ON CONFLICT(file_id, stage) DO UPDATE SET params_hash=excluded.params_hash, "
            "status='running', attempts=file_stages.attempts+1, error=NULL, updated_at=excluded.updated_at",
            (file_id, stage, params_hash, utcnow()),
        )
        self.conn.commit()

    def finish_stage(
        self,
        file_id: str,
        stage: str,
        status: str,
        params_hash: str,
        error: Optional[str] = None,
        stats: Optional[dict] = None,
    ) -> None:
        self.execute(
            "UPDATE file_stages SET status=?, params_hash=?, error=?, stats_json=?, updated_at=? "
            "WHERE file_id=? AND stage=?",
            (status, params_hash, error, dumps(stats) if stats is not None else None,
             utcnow(), file_id, stage),
        )
        self.conn.commit()

    def stage_is_current(self, file_id: str, stage: str, params_hash: str) -> bool:
        row = self.get_stage(file_id, stage)
        return bool(row and row["status"] in ("done", "partial") and row["params_hash"] == params_hash)

    # ---------- segments ----------
    def replace_segments(self, file_id: str, vad_hash: str, rows: list[dict]) -> None:
        # 一个文件的切片永远来自同一套 VAD 参数；参数变更后旧哈希段必须一起清掉
        self.execute("DELETE FROM segments WHERE file_id=?", (file_id,))
        now = utcnow()
        self.executemany(
            "INSERT INTO segments(segment_id, file_id, vad_hash, seq, start_s, end_s, "
            "duration_s, wav_path, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (r["segment_id"], file_id, vad_hash, r["seq"], r["start_s"], r["end_s"],
                 r["duration_s"], r["wav_path"], now)
                for r in rows
            ],
        )
        self.conn.commit()

    def segments(self, file_id: str, vad_hash: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM segments WHERE file_id=? AND vad_hash=? ORDER BY seq",
            (file_id, vad_hash),
        )

    def pending_segments(
        self, file_id: str, stage: str, params_hash: str, vad_hash: str | None = None
    ) -> list[sqlite3.Row]:
        state_col = f"{stage}_state"
        hash_col = f"{stage}_params_hash"
        sql = (
            f"SELECT * FROM segments WHERE file_id=? AND "
            f"(COALESCE({state_col},'pending') != 'done' OR IFNULL({hash_col},'') != ?)"
        )
        args: list = [file_id, params_hash]
        if vad_hash:
            sql += " AND vad_hash=?"
            args.append(vad_hash)
        return self.query(sql + " ORDER BY seq", tuple(args))

    def update_sv(
        self,
        segment_id: str,
        params_hash: str,
        state: str,
        embedding: Optional[bytes] = None,
        score: Optional[float] = None,
        template_idx: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self.execute(
            "UPDATE segments SET sv_state=?, sv_params_hash=?, sv_emb=?, sv_score=?, "
            "sv_tidx=?, error=?, updated_at=? WHERE segment_id=?",
            (state, params_hash, embedding, score, template_idx, error, utcnow(), segment_id),
        )

    def update_asr(
        self,
        segment_id: str,
        params_hash: str,
        state: str,
        text: Optional[str] = None,
        raw: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.execute(
            "UPDATE segments SET asr_state=?, asr_params_hash=?, asr_text=?, asr_raw=?, "
            "error=?, updated_at=? WHERE segment_id=?",
            (state, params_hash, text, raw, error, utcnow(), segment_id),
        )

    def update_noise(
        self,
        segment_id: str,
        params_hash: str,
        state: str,
        prob: Optional[float] = None,
        flag: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        self.execute(
            "UPDATE segments SET noise_state=?, noise_params_hash=?, noisy_prob=?, "
            "noisy_flag=?, error=?, updated_at=? WHERE segment_id=?",
            (state, params_hash, prob, flag, error, utcnow(), segment_id),
        )

    def segments_for_file(self, file_id: str, vad_hash: str) -> list[sqlite3.Row]:
        return self.segments(file_id, vad_hash)

    # ---------- exports ----------
    def set_export(self, row: dict) -> None:
        self.execute(
            "INSERT INTO exports(file_id, status, artifact_path, artifact_sha256, threshold, "
            "export_params_hash, segment_count, skipped_count, generated_at, error) "
            "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET "
            "status=excluded.status, artifact_path=excluded.artifact_path, "
            "artifact_sha256=excluded.artifact_sha256, "
            "threshold=excluded.threshold, export_params_hash=excluded.export_params_hash, "
            "segment_count=excluded.segment_count, skipped_count=excluded.skipped_count, "
            "generated_at=excluded.generated_at, error=excluded.error",
            (row["file_id"], row["status"], row.get("artifact_path"),
             row.get("artifact_sha256"), row.get("threshold"),
             row.get("export_params_hash"), row.get("segment_count", 0),
             row.get("skipped_count", 0), row.get("generated_at"), row.get("error")),
        )
        self.conn.commit()

    def get_export(self, file_id: str) -> Optional[sqlite3.Row]:
        return self.one("SELECT * FROM exports WHERE file_id=?", (file_id,))

    def counts(self, file_id: str, vad_hash: str, threshold: float = 0.4) -> dict:
        row = self.one(
            "SELECT COUNT(*) AS total, "
            "SUM(sv_state='done') AS sv_done, "
            "SUM(asr_state='done') AS asr_done, "
            "SUM(noise_state='done') AS noise_done, "
            "SUM(sv_score >= ?) AS kept "
            "FROM segments WHERE file_id=? AND vad_hash=?",
            (threshold, file_id, vad_hash),
        )
        return {k: (row[k] or 0) for k in ("total", "sv_done", "asr_done", "noise_done", "kept")}
