from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicedata import export as export_mod
from voicedata.db import Database
from voicedata.pipeline import Pipeline
from voicedata.validate import validate_job
from voicedata.workcache import cleanup_file_audio


class Counter:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def detect(self, *a, **k):
        self.calls += 1
        return self.inner.detect(*a, **k)

    def frame_probs(self, *a, **k):
        self.calls += 1
        return self.inner.frame_probs(*a, **k)

    def embed(self, *a, **k):
        self.calls += 1
        return self.inner.embed(*a, **k)

    def transcribe(self, *a, **k):
        self.calls += 1
        return self.inner.transcribe(*a, **k)

    def classify(self, *a, **k):
        self.calls += 1
        return self.inner.classify(*a, **k)


def _counters(pipe: Pipeline) -> dict[str, Counter]:
    wrapped = {k: Counter(v) for k, v in pipe.adapters.items()}
    pipe.adapters = wrapped
    return wrapped


def _out_zip(cfg) -> Path:
    return cfg.job_dir("output", "full") / "直播 测试.zip"


def _read_export(cfg):
    with zipfile.ZipFile(_out_zip(cfg)) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        body = zf.read("segments.jsonl").decode("utf-8")
        segs = [json.loads(line) for line in body.splitlines() if line.strip()]
        audio = {name: zf.read(name) for name in zf.namelist()
                 if name.startswith("audio/")}
    return manifest, segs, audio


def test_end_to_end_and_validate(wav_input):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    out = _out_zip(cfg)
    assert out.exists()
    manifest, segs, audio = _read_export(cfg)
    assert manifest["audio_file"] == "直播 测试.wav"
    assert manifest["format"] == "flac-zip"
    assert manifest["audio_format"] == "flac/16000/1/16bit"
    assert manifest["segment_count"] == len(segs) == 2
    assert [round(s["start_s"], 2) for s in segs] == [0.5, 3.5]
    assert all(s["asr_text"].startswith("测试文本") for s in segs)
    assert all(s["noisy"] in (0, 1) for s in segs)
    assert all(0.99 <= s["sv_score"] <= 1.0 for s in segs)
    assert set(audio) == {s["audio"] for s in segs}
    for raw in audio.values():
        info = sf.info(io.BytesIO(raw))
        assert (info.channels, info.samplerate, info.subtype) == (1, 16000, "PCM_16")

    out_dir = cfg.job_dir("output", "full")
    lines = [json.loads(x) for x in
             (out_dir / "index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert {x["archive"] for x in lines} == {out.name}
    assert all(x["audio_file"] == "直播 测试.wav" for x in lines)
    meta = json.loads((out_dir / "index.meta.json").read_text(encoding="utf-8"))
    assert (meta["file_count"], meta["segment_count"]) == (1, 2)

    report = validate_job(cfg, "full")
    assert report["ok"]
    assert report["ok_count"] == report["file_count"] == 1
    assert report["total_segments"] == 2
    assert report["files"][0]["errors"] == []
    assert report["index"]["ok"] and report["index"]["segments"] == 2

    work = cfg.job_dir("work", "full")
    assert (work / "state.db").exists()
    assert not list((work / "audio").glob("*.wav"))
    assert not list((work / "segments").rglob("*.wav"))


def test_aed_drops_all_singing_segments(wav_input, monkeypatch):
    from voicedata.adapters.fakes import FakeAED

    cfg, _ = wav_input
    # AED 判整段都是歌声：VAD 段全部被剔除，导出空段清单也应是合法结果
    monkeypatch.setattr(
        FakeAED, "frame_probs",
        lambda self, wav: np.zeros((max(1, len(wav) // 160), 3), dtype=np.float32),
    )
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    manifest, segs, audio = _read_export(cfg)
    assert manifest["segment_count"] == 0
    assert segs == [] and audio == {}
    out_dir = cfg.job_dir("output", "full")
    assert (out_dir / "index.jsonl").read_text(encoding="utf-8") == ""
    report = validate_job(cfg, "full")
    assert report["ok"]
    assert report["ok_count"] == report["file_count"] == 1
    assert report["total_segments"] == 0


def test_rerun_is_idempotent(wav_input):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()
    out = _out_zip(cfg)
    index = cfg.job_dir("output", "full") / "index.jsonl"
    before, before_index = out.read_bytes(), index.read_bytes()

    with Pipeline(cfg, "full") as pipe:
        counters = _counters(pipe)
        pipe.run()
        assert all(c.calls == 0 for c in counters.values()), {
            k: c.calls for k, c in counters.items()}
    assert out.read_bytes() == before
    assert index.read_bytes() == before_index


def test_threshold_reexport_without_models(wav_input):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()
    old_manifest, old_segs, _ = _read_export(cfg)
    work = cfg.job_dir("work", "full")
    assert not list((work / "segments").rglob("*.wav"))

    with Pipeline(cfg, "full") as pipe:
        counters = _counters(pipe)
        pipe.run(stages=["export"], threshold=0.5)
        assert all(c.calls == 0 for c in counters.values())
    manifest, segs, _ = _read_export(cfg)
    assert manifest["target_threshold"] == 0.5
    assert manifest["segment_count"] == old_manifest["segment_count"]
    # 重新导出会按需恢复片段，导出后再次清理
    assert not list((work / "segments").rglob("*.wav"))

    with Pipeline(cfg, "full") as pipe:
        pipe.run(stages=["export"], threshold=0.4)
    manifest, segs, _ = _read_export(cfg)
    assert manifest["target_threshold"] == 0.4
    assert len(segs) == len(old_segs)


def test_cleanup_can_be_disabled(wav_input):
    cfg, _ = wav_input
    cfg.apply_overrides({"run.cleanup_audio_after_export": False})
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    db = Database(cfg.job_dir("work", "full") / "state.db", readonly=True)
    row = db.files()[0]
    assert Path(row["decoded_wav"]).exists()
    segs = db.query("SELECT wav_path FROM segments WHERE file_id=?", (row["file_id"],))
    assert segs and all(Path(s["wav_path"]).exists() for s in segs)
    db.close()


def test_cleanup_does_not_touch_files_outside_work(wav_input, tmp_path):
    cfg, _ = wav_input
    cfg.apply_overrides({"run.cleanup_audio_after_export": False})
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    outside = tmp_path / "keep.wav"
    outside.write_bytes(b"not under work")
    db = Database(cfg.job_dir("work", "full") / "state.db")
    fid = db.files()[0]["file_id"]
    db.execute("UPDATE files SET decoded_wav=? WHERE file_id=?", (str(outside), fid))
    db.commit()

    cleanup_file_audio(db, cfg, "full", fid)
    assert outside.exists()
    db.close()


def test_export_failure_keeps_previous_zip(wav_input, monkeypatch):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()
    out = _out_zip(cfg)
    before = out.read_bytes()

    def boom(_):
        raise RuntimeError("注入的编码失败")

    monkeypatch.setattr(export_mod, "wav_to_flac_bytes", boom)
    with Pipeline(cfg, "full") as pipe:
        pipe.run(stages=["export"], force=True)
    assert out.read_bytes() == before
    db = Database(cfg.job_dir("work", "full") / "state.db")
    row = db.one("SELECT * FROM exports")
    assert row["status"] == "failed"
    stage = db.get_stage(row["file_id"], "export")
    assert stage["status"] == "failed"
    db.close()

    monkeypatch.undo()
    with Pipeline(cfg, "full") as pipe:
        pipe.run(stages=["export"], force=True)
    assert validate_job(cfg, "full")["ok"]


def test_json_compat_format(wav_input):
    cfg, _ = wav_input
    cfg.apply_overrides({"export.format": "json"})
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    out_dir = cfg.job_dir("output", "full")
    out = out_dir / "直播 测试.json"
    assert out.exists()
    assert not list(out_dir.glob("*.zip"))
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["format"] == "json-base64"
    assert doc["audio_format"] == "wav/pcm_s16le/16000/1"
    assert doc["segment_count"] == len(doc["segments"]) == 2
    assert all("audio_base64" in s for s in doc["segments"])
    report = validate_job(cfg, "full")
    assert report["ok"]
    assert report["index"]["path"] is None


def test_validate_detects_missing_index(wav_input):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()
    (cfg.job_dir("output", "full") / "index.jsonl").unlink()
    report = validate_job(cfg, "full")
    assert not report["ok"]
    assert any("index.jsonl" in msg for msg in report["errors"])


def test_resume_after_interrupted_asr(wav_input):
    cfg, _ = wav_input
    with Pipeline(cfg, "full") as pipe:
        pipe.run()

    db = Database(cfg.job_dir("work", "full") / "state.db")
    db.execute("UPDATE segments SET asr_state='running'")
    db.execute("UPDATE file_stages SET status='running' WHERE stage='asr'")
    db.commit()
    db.close()

    with Pipeline(cfg, "full") as pipe:
        counters = _counters(pipe)
        pipe.run()
        assert counters["asr"].calls > 0
        assert counters["vad"].calls == 0
        assert counters["sv"].calls == 0
        assert counters["noise"].calls == 0
    assert validate_job(cfg, "full")["ok"]
