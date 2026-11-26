"""输入文件发现与登记。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .audio import probe_duration
from .db import Database

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}


class IngestError(RuntimeError):
    pass


def file_id_for(source_key: str) -> str:
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]


def discover(cfg, pattern: str | None = None) -> list[Path]:
    root = cfg.path("paths.input_dir")
    if not root.exists():
        raise IngestError(f"输入目录不存在：{root}")
    found = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("._"):
            continue
        if p.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if p.stat().st_size == 0:
            continue
        if pattern and pattern.lower() not in p.name.lower():
            continue
        found.append(p)
    if pattern and not found:
        raise IngestError(f"没有匹配 --file 的音频：{pattern}")
    if not found:
        raise IngestError(f"输入目录没有可用音频：{root}")
    return found


def register(
    db: Database,
    cfg,
    pattern: str | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> list[str]:
    paths = discover(cfg, pattern)
    excerpt = start_s is not None or duration_s is not None
    if excerpt and len(paths) != 1:
        raise IngestError("使用 --start/--duration 时必须恰好匹配一个文件（配合 --file）")
    root = cfg.path("paths.input_dir")
    offsets: list[float] = []
    durations: list[float | None] = []
    if excerpt:
        offsets = [float(start_s or 0.0)]
        durations = [float(duration_s) if duration_s else None]
        if offsets[0] < 0:
            raise IngestError("--start 不能为负")
        if durations[0] is not None and durations[0] <= 0:
            raise IngestError("--duration 必须为正")
    else:
        offsets = [0.0] * len(paths)
        durations = [None] * len(paths)

    ids: list[str] = []
    for path, offset, dur in zip(paths, offsets, durations):
        rel = path.relative_to(root).as_posix()
        if dur is None and offset == 0:
            source_key = rel
        else:
            source_key = f"{rel}#offset={offset:.3f}&dur={dur if dur is not None else 'end'}"
        fid = file_id_for(source_key)
        total = probe_duration(path)
        if offset >= total:
            raise IngestError(f"--start={offset} 超出音频时长 {total:.1f}s：{path.name}")
        db.upsert_file(
            file_id=fid,
            source_key=source_key,
            rel_path=rel,
            audio_file=path.name,
            stem=path.stem,
            size_bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,
            source_duration_s=total,
            offset_s=offset,
            excerpt_duration_s=dur,
        )
        ids.append(fid)

    stems = {}
    for fid in ids:
        row = db.get_file(fid)
        stems.setdefault(row["stem"], []).append(row["audio_file"])
    collisions = {stem: names for stem, names in stems.items() if len(names) > 1}
    if collisions:
        raise IngestError(f"存在同名输出（stem 冲突）：{collisions}")
    return ids
