"""音频解码、切片、WAV/FLAC 编解码与时间工具。"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(RuntimeError):
    pass


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg 失败（{proc.returncode}）：{proc.stderr.strip()[:800]}")


def probe_duration(path: str | Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise AudioError(f"ffprobe 失败：{proc.stderr.strip()[:800]}")
    try:
        dur = float(json.loads(proc.stdout)["format"]["duration"])
    except Exception as exc:
        raise AudioError(f"无法读取音频时长：{exc}") from exc
    if not np.isfinite(dur) or dur <= 0:
        raise AudioError(f"音频时长为 0 或非法：{dur}")
    return dur


def decode_to_wav(
    src: str | Path,
    dst: str | Path,
    sample_rate: int = 16000,
    offset_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    args: list[str] = []
    if offset_s and offset_s > 0:
        args += ["-ss", f"{offset_s:.3f}"]
    if duration_s and duration_s > 0:
        args += ["-t", f"{duration_s:.3f}"]
    args += [
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", "-f", "wav", str(tmp),
    ]
    try:
        run_ffmpeg(args)
        if not tmp.exists() or tmp.stat().st_size <= 44:
            raise AudioError("ffmpeg 未产生有效音频")
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def wav_info(path: str | Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {
            "channels": w.getnchannels(),
            "sample_rate": w.getframerate(),
            "sample_width": w.getsampwidth(),
            "frames": w.getnframes(),
            "duration_s": w.getnframes() / float(w.getframerate()),
        }


def read_int16(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        total = len(f)
        start = max(0, int(round(start_s * sr)))
        stop = total if end_s is None else min(total, int(round(end_s * sr)))
        f.seek(start)
        data = f.read(stop - start, dtype="int16", always_2d=False)
    return np.asarray(data, dtype=np.int16)


def read_float32(path: str | Path, start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        total = len(f)
        start = max(0, int(round(start_s * sr)))
        stop = total if end_s is None else min(total, int(round(end_s * sr)))
        f.seek(start)
        data = f.read(stop - start, dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32)


def slice_wav(src: str | Path, dst: str | Path, start_s: float, end_s: float) -> dict:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = read_int16(src, start_s, end_s)
    if data.size == 0:
        raise AudioError(f"切片为空：{start_s:.3f}-{end_s:.3f}")
    sf.write(str(dst), data, 16000, subtype="PCM_16", format="WAV")
    info = wav_info(dst)
    return info


def wav_base64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def wav_to_flac_bytes(path: str | Path) -> bytes:
    """把 16k/单声道/16bit WAV 转成 FLAC 字节，供 zip 直接写入。"""
    data, sr = sf.read(str(path), dtype="int16", always_2d=False)
    data = np.asarray(data, dtype=np.int16)
    if data.ndim != 1:
        raise AudioError(f"切片不是单声道：{path}")
    if sr != 16000:
        raise AudioError(f"切片采样率不是 16000：{path}（{sr}）")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def flac_info(raw: bytes) -> dict:
    """读取 FLAC 字节的格式与时长，不落盘。"""
    with sf.SoundFile(io.BytesIO(raw)) as f:
        return {
            "channels": f.channels,
            "sample_rate": f.samplerate,
            "subtype": f.subtype,
            "frames": len(f),
            "duration_s": len(f) / float(f.samplerate) if f.samplerate else 0.0,
        }


def flac_to_wav_base64(raw: bytes) -> str:
    """把 zip 里的 FLAC 转成 WAV base64；校准页用浏览器直接播放。"""
    data, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, np.asarray(data, dtype=np.int16), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_wav_header(path: str | Path) -> None:
    info = wav_info(path)
    if info["channels"] != 1 or info["sample_rate"] != 16000 or info["sample_width"] != 2:
        raise AudioError(f"WAV 格式不符合 16k/单声道/16bit：{info}")
