"""FireRedVAD 分段、长音频分窗与片段切片。"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from .audio import read_int16, slice_wav
from .db import Database
from .decode import run_decode

logger = logging.getLogger(__name__)


def plan_windows(duration_s: float, window_s: float, context_s: float):
    """返回 [(core_start, core_end, win_start, win_end)]。

    上下文长度不小于最大段长，保证中点落在核心区的段完整出现在窗口里。
    """
    windows = []
    start = 0.0
    while start < duration_s - 1e-6:
        core_end = min(start + window_s, duration_s)
        win_start = max(0.0, start - context_s)
        win_end = min(duration_s, core_end + context_s)
        windows.append((start, core_end, win_start, win_end))
        start = core_end
    return windows


def refine_runs(
    probs,
    duration_s: float,
    *,
    speech_threshold: float = 0.4,
    min_speech_s: float = 0.3,
    merge_gap_s: float = 0.8,
    pad_s: float = 0.05,
    smooth_frames: int = 5,
    singing_threshold: float = 0.5,
    max_merged_s: float = 12.0,
) -> list[tuple[float, float]]:
    """从 AED 逐帧概率里取出 speech 帧，返回相对片段的 (start, end) 列表。

    probs 形状 (T,3)，列为 [speech, singing, music]；帧率由 T/duration_s 推出。
    先中值平滑抑制抖动，再按阈值取连续帧；间隔不超过 merge_gap_s、间隔内歌声概率低于
    singing_threshold、且合并后不超过 max_merged_s 的 run 会合并（避免把 BGM 歌词拼进来）。
    然后两侧补 pad，最后丢弃过短段。
    """
    if duration_s <= 0 or probs is None:
        return []
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    if arr.shape[1] < 2:
        raise ValueError(f"AED 概率应为 (T,3)，实际 {arr.shape}")
    speech = arr[:, 0]
    singing = arr[:, 1]
    if smooth_frames and smooth_frames > 1:
        speech = median_filter(speech, size=int(smooth_frames), mode="nearest")
    fps = len(speech) / float(duration_s)
    if fps <= 0:
        return []

    on = speech >= float(speech_threshold)
    runs: list[list[float]] = []  # [start_s, end_s, first_frame, last_frame]
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(on) and on[j + 1]:
            j += 1
        runs.append([i / fps, (j + 1) / fps, i, j])
        i = j + 1

    merged: list[list[float]] = []
    for start, end, i0, i1 in runs:
        if merged:
            prev = merged[-1]
            gap_frames = singing[int(prev[3]) + 1:i0]
            gap_clean = gap_frames.size == 0 or float(gap_frames.max()) < float(singing_threshold)
            if (
                start - prev[1] <= merge_gap_s
                and gap_clean
                and end - prev[0] <= max_merged_s
            ):
                prev[1] = max(prev[1], end)
                prev[3] = i1
                continue
        merged.append([start, end, i0, i1])

    out: list[tuple[float, float]] = []
    for start, end, _, _ in merged:
        start = max(0.0, start - pad_s)
        end = min(float(duration_s), end + pad_s)
        if end - start < min_speech_s:
            continue
        # pad 可能让本来不相邻的 run 重叠（merge_gap_ms < 2*pad_ms 时），合并保证不重叠
        if out and start <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return [(round(s, 3), round(e, 3)) for s, e in out]


def merge_adjacent(
    segs: list[tuple[float, float]],
    gap_s: float,
    max_s: float,
    gap_ok=None,
) -> list[tuple[float, float]]:
    """合并相邻区间：间隔不超过 gap_s、合并后不超过 max_s 且 gap_ok(prev_end, next_start) 允许。

    gap_ok 用于拒绝跨越歌声等非语音内容的合并；重叠区间直接合并，不调用 gap_ok。
    """
    out: list[tuple[float, float]] = []
    for start, end in segs:
        if out:
            prev_start, prev_end = out[-1]
            gap = start - prev_end
            if gap <= gap_s and end - prev_start <= max_s:
                if gap <= 1e-6 or gap_ok is None or gap_ok(prev_end, start):
                    out[-1] = (prev_start, max(prev_end, end))
                    continue
        out.append((start, end))
    return out


def _cleanup_old_segment_dirs(seg_base: Path, file_id: str, keep_hash: str) -> None:
    """VAD 参数变更后，删掉该文件在旧哈希目录下的切片。"""
    try:
        for d in seg_base.iterdir():
            if not d.is_dir() or d.name == keep_hash:
                continue
            target = d / file_id
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                logger.info("清理旧切片目录：%s", target)
            try:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass
    except OSError as exc:
        logger.warning("清理旧切片目录失败：%s", exc)


def detect_file(adapter, wav_path: Path, duration_s: float, window_s: float, context_s: float):
    seen: set[tuple[float, float]] = set()
    found: list[tuple[float, float]] = []
    for core_start, core_end, win_start, win_end in plan_windows(duration_s, window_s, context_s):
        audio = read_int16(wav_path, win_start, win_end)
        if audio.size < 1600:
            continue
        for s, e in adapter.detect(audio):
            s_abs = win_start + s
            e_abs = win_start + e
            mid = (s_abs + e_abs) / 2.0
            in_core = core_start - 1e-6 <= mid < core_end - 1e-6
            if not in_core and core_end >= duration_s - 1e-6 and mid <= core_end + 1e-6:
                in_core = True
            if not in_core:
                continue
            s_abs = max(0.0, min(duration_s, s_abs))
            e_abs = max(0.0, min(duration_s, e_abs))
            if e_abs <= s_abs:
                continue
            key = (round(s_abs, 3), round(e_abs, 3))
            if key in seen:
                continue
            seen.add(key)
            found.append((s_abs, e_abs))
    found.sort()
    merged: list[tuple[float, float]] = []
    for s, e in found:
        if merged and s <= merged[-1][1] + 1e-3:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def segment_id_for(file_id: str, vad_hash: str, seq: int) -> str:
    return hashlib.sha1(f"{file_id}|{vad_hash}|{seq}".encode("utf-8")).hexdigest()[:16]


def run_vad(db: Database, cfg, job: str, file_ids: list[str], adapters, force: bool = False) -> None:
    params = cfg.stage_hash("vad", job)
    window_s = float(cfg.get("vad.window_s", 600))
    context_s = float(cfg.get("vad.context_s", 20))
    min_seg_s = float(cfg.get("vad.min_speech_ms", 350)) / 1000.0
    max_seg_s = float(cfg.get("vad.max_segment_s", 20.0))
    aed_enabled = bool(cfg.get("aed.enabled", True))
    aed_thr = float(cfg.get("aed.speech_threshold", 0.4))
    aed_min_s = float(cfg.get("aed.min_speech_ms", 300)) / 1000.0
    aed_gap_s = float(cfg.get("aed.merge_gap_ms", 800)) / 1000.0
    aed_pad_s = float(cfg.get("aed.pad_ms", 50)) / 1000.0
    aed_sing_thr = float(cfg.get("aed.singing_threshold", 0.5))
    aed_max_merged_s = float(cfg.get("aed.max_merged_s", 12.0))
    sample_rate = int(cfg.get("decode.sample_rate", 16000))
    seg_base = cfg.job_dir("work", job) / "segments"

    for fid in file_ids:
        row = db.get_file(fid)
        if row is None:
            continue
        # 已完成且片段记录完整时直接跳过，不因为 work 音频被清理而重新解码
        if not force and db.stage_is_current(fid, "vad", params):
            existing = db.segments(fid, params)
            stats = db.get_stage(fid, "vad")
            expected = None
            if stats and stats["stats_json"]:
                try:
                    import json

                    expected = json.loads(stats["stats_json"]).get("segment_count")
                except Exception:
                    expected = None
            if expected is None or len(existing) == expected:
                continue

        if (
            row["decode_state"] != "done"
            or not row["decoded_wav"]
            or not Path(row["decoded_wav"]).exists()
        ):
            # 导出后 work 音频会被清理；需要重跑 vad 时先按需解码
            logger.warning("VAD 需要先解码，正在补齐：%s", row["audio_file"])
            run_decode(db, cfg, job, [fid], force=True)
            row = db.get_file(fid)
        if (
            row is None
            or row["decode_state"] != "done"
            or not row["decoded_wav"]
            or not Path(row["decoded_wav"]).exists()
        ):
            logger.warning("跳过 VAD（解码失败）：%s", row["audio_file"] if row else fid)
            db.begin_stage(fid, "vad", params)
            db.finish_stage(fid, "vad", "skipped", params, error="decode 未完成或音频已清理")
            continue

        db.begin_stage(fid, "vad", params)
        try:
            wav_path = Path(row["decoded_wav"])
            duration = float(row["decoded_duration_s"] or 0.0)
            if duration <= 0.05:
                db.replace_segments(fid, params, [])
                _cleanup_old_segment_dirs(seg_base, fid, params)
                db.finish_stage(fid, "vad", "done", params, stats={
                    "raw_segment_count": 0, "raw_speech_s": 0.0,
                    "trimmed_s": 0.0, "segment_count": 0, "speech_s": 0,
                })
                continue
            offset = float(row["offset_s"] or 0.0)
            raw_segments = detect_file(adapters["vad"], wav_path, duration, window_s, context_s)
            raw_speech_s = sum(le - ls for ls, le in raw_segments)
            trimmed_s = 0.0
            if aed_enabled:
                local_segments: list[tuple[float, float]] = []
                for ls, le in raw_segments:
                    audio = read_int16(wav_path, ls, le)
                    if audio.size == 0:
                        continue
                    probs = adapters["aed"].frame_probs(audio)
                    audio_dur = audio.size / float(sample_rate)
                    runs = refine_runs(
                        probs, audio_dur,
                        speech_threshold=aed_thr, min_speech_s=aed_min_s,
                        merge_gap_s=aed_gap_s, pad_s=aed_pad_s,
                        singing_threshold=aed_sing_thr, max_merged_s=aed_max_merged_s,
                    )
                    local_segments.extend((ls + s, ls + e) for s, e in runs)

                def gap_ok(gap_start: float, gap_end: float) -> bool:
                    # 跨原始 FireRedVAD 段的间隔：独立跑一次 AED，间隔里有歌声就不合并
                    if gap_end - gap_start <= 0.05:
                        return True
                    gap_audio = read_int16(wav_path, gap_start, gap_end)
                    if gap_audio.size == 0:
                        return True
                    gap_probs = np.asarray(adapters["aed"].frame_probs(gap_audio), dtype=np.float32)
                    if gap_probs.ndim != 2 or gap_probs.shape[0] == 0 or gap_probs.shape[1] < 2:
                        return True
                    return float(gap_probs[:, 1].max()) < aed_sing_thr

                local_segments = merge_adjacent(
                    local_segments, aed_gap_s, aed_max_merged_s, gap_ok,
                )
                kept_s = sum(e - s for s, e in local_segments)
                trimmed_s = max(0.0, raw_speech_s - kept_s)
            else:
                local_segments = raw_segments

            seg_dir = seg_base / params / fid
            seg_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            speech_s = 0.0
            for seq, (ls, le) in enumerate(local_segments):
                if not aed_enabled and le - ls < min_seg_s:
                    continue
                if le - ls > max_seg_s + 2.0:
                    logger.warning("段过长（%.1fs），保留：%s", le - ls, row["audio_file"])
                abs_start = offset + ls
                abs_end = offset + le
                sid = segment_id_for(fid, params, seq)
                wav_out = seg_dir / f"{seq:05d}_{sid}.wav"
                info = slice_wav(wav_path, wav_out, ls, le)
                rows.append({
                    "segment_id": sid,
                    "seq": seq,
                    "start_s": round(abs_start, 3),
                    "end_s": round(abs_end, 3),
                    "duration_s": round(abs_end - abs_start, 3),
                    "wav_path": str(wav_out),
                })
                speech_s += info["duration_s"]
            db.replace_segments(fid, params, rows)
            _cleanup_old_segment_dirs(seg_base, fid, params)
            db.finish_stage(fid, "vad", "done", params, stats={
                "raw_segment_count": len(raw_segments),
                "raw_speech_s": round(raw_speech_s, 2),
                "trimmed_s": round(trimmed_s, 2),
                "segment_count": len(rows),
                "speech_s": round(speech_s, 2),
            })
            logger.info("VAD 完成：%s，%d 段（原始 %d 段，剔除 %.1fs），%.1fs 语音",
                        row["audio_file"], len(rows), len(raw_segments), trimmed_s, speech_s)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("VAD 失败：%s：%s", row["audio_file"], msg)
            db.finish_stage(fid, "vad", "failed", params, error=msg)
