"""work 目录音频缓存：导出后清理，重跑阶段时按需恢复。"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .audio import decode_to_wav
from .db import Database

logger = logging.getLogger(__name__)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _prune_empty(parent: Path, root: Path) -> None:
    for current in (parent, *parent.parents):
        if current == root or not _under(current, root):
            break
        try:
            current.rmdir()
        except OSError:
            break


def cleanup_file_audio(db: Database, cfg, job: str, file_id: str) -> int:
    """删除某个文件的整段解码 WAV 与所有候选片段 WAV，返回释放的字节数。

    只动 work 目录下的音频文件；state.db、日志和 out/ 下的成品不受影响。
    """
    work = cfg.job_dir("work", job).resolve()
    row = db.get_file(file_id)
    if row is None:
        return 0

    targets: list[Path] = []
    if row["decoded_wav"]:
        targets.append(Path(row["decoded_wav"]))
    seg_root = work / "segments"
    if seg_root.exists():
        targets.extend(seg_root.glob(f"*/{file_id}"))
    for seg in db.query("SELECT wav_path FROM segments WHERE file_id=?", (file_id,)):
        if seg["wav_path"]:
            targets.append(Path(seg["wav_path"]))

    freed = 0
    parents: set[Path] = set()
    for target in targets:
        if not _under(target, work):
            logger.warning("跳过 work 目录之外的缓存文件：%s", target)
            continue
        try:
            if target.is_dir():
                freed += _dir_size(target)
                parents.add(target.parent)
                shutil.rmtree(target)
            elif target.is_file():
                freed += target.stat().st_size
                parents.add(target.parent)
                target.unlink()
        except OSError as exc:
            logger.warning("清理缓存失败：%s：%s", target, exc)

    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        _prune_empty(parent, work)

    if freed:
        logger.info("清理工作音频：%s，释放 %.1f MB",
                    row["audio_file"], freed / (1024 * 1024))
    return freed


def ensure_segment_audio(db: Database, cfg, job: str, file_id: str, rows) -> int:
    """确保 rows 里的片段 WAV 存在；缺失时按原始时间从源音频重新切出。

    rows 可以是数据库行，也可以是 (row, ...) 元组。正常跑完导出后 work 里不再保留
    片段；之后只改导出阈值或重跑声纹/ASR/噪音时，用这个函数按需恢复。
    """
    items = [item[0] if isinstance(item, tuple) else item for item in rows]
    missing = [
        row for row in items
        if not row["wav_path"] or not Path(row["wav_path"]).exists()
    ]
    if not missing:
        return 0

    frow = db.get_file(file_id)
    if frow is None:
        raise RuntimeError(f"找不到文件记录：{file_id}")
    src = cfg.path("paths.input_dir") / frow["rel_path"]
    if not src.exists():
        raise FileNotFoundError(
            f"源音频不存在，无法恢复片段：{src}；work 中的音频已清理，"
            "请恢复源文件后重跑，或直接使用 out/ 中的成品")
    sample_rate = int(cfg.get("decode.sample_rate", 16000))
    work = cfg.job_dir("work", job).resolve()

    restored = 0
    for row in missing:
        start = float(row["start_s"])
        end = float(row["end_s"])
        if end <= start:
            raise ValueError(f"片段时长非法：{row['segment_id']} {start}->{end}")
        wav_path = row["wav_path"]
        if not wav_path:
            wav_path = str(work / "segments" / "restored" / file_id /
                           f"{row['segment_id']}.wav")
            db.execute("UPDATE segments SET wav_path=? WHERE segment_id=?",
                       (wav_path, row["segment_id"]))
        target = Path(wav_path)
        if not _under(target, work):
            raise RuntimeError(f"片段缓存路径不在 work 下：{target}")
        decode_to_wav(src, target, sample_rate=sample_rate,
                      offset_s=start, duration_s=end - start)
        restored += 1

    db.commit()
    logger.info("恢复片段音频：%s，%d 个", frow["audio_file"], restored)
    return restored
