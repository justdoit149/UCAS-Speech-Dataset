from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicedata.adapters.fakes import FakeVAD
from voicedata.audio import wav_base64
from voicedata.db import Database
from voicedata.export import _is_filler, _normalize_text, _skip_reason
from voicedata.ingest import IngestError, discover, register
from voicedata.sv import load_templates, score_embeddings
from voicedata.vad import detect_file, merge_adjacent, plan_windows, refine_runs


def test_plan_windows_covers_whole_file():
    wins = plan_windows(1500.0, 600.0, 20.0)
    assert [w[0] for w in wins] == [0.0, 600.0, 1200.0]
    assert wins[-1][1] == 1500.0
    for core_start, core_end, win_start, win_end in wins:
        assert core_start <= core_end
        assert win_start <= core_start and win_end >= core_end
    # 相邻核心区无缝衔接
    for a, b in zip(wins, wins[1:]):
        assert a[1] == b[0]
    # 上下文不短于最大段长 20s
    assert wins[0][2] == 0.0 and wins[1][2] == 580.0


def test_plan_windows_empty():
    assert plan_windows(0.0, 600.0, 20.0) == []


def test_detect_file_returns_sorted_unique(tmp_path: Path):
    sr = 16000
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros(sr * 6, dtype=np.int16), sr, subtype="PCM_16")
    segs = detect_file(FakeVAD(None), p, 6.0, window_s=2.0, context_s=2.0)
    assert segs == sorted(segs)
    assert all(0 <= s < e <= 6.0 for s, e in segs)
    assert len(set(segs)) == len(segs)
    for a, b in zip(segs, segs[1:]):
        assert a[1] <= b[0] + 1e-6


def test_refine_runs_keeps_speech_runs():
    probs = np.zeros((1000, 3), dtype=np.float32)
    probs[100:200, 0] = 1.0
    probs[240:400, 0] = 1.0
    probs[800:810, 0] = 1.0  # 0.2s，过短，应丢弃
    runs = refine_runs(probs, 10.0, speech_threshold=0.4, min_speech_s=0.3,
                       merge_gap_s=0.3, pad_s=0.05)
    assert len(runs) == 2
    assert runs[0] == pytest.approx((0.95, 2.05))
    assert runs[1] == pytest.approx((2.35, 4.05))


def test_refine_runs_merges_short_gaps_and_clamps():
    probs = np.zeros((1000, 3), dtype=np.float32)
    probs[0:100, 0] = 1.0
    probs[120:300, 0] = 1.0
    runs = refine_runs(probs, 10.0, merge_gap_s=0.3, pad_s=0.05, min_speech_s=0.3)
    assert len(runs) == 1
    assert runs[0] == pytest.approx((0.0, 3.05))


def test_refine_runs_pad_overlap_does_not_overlap():
    probs = np.zeros((1000, 3), dtype=np.float32)
    probs[100:150, 0] = 1.0  # 1.0–1.5s
    probs[165:215, 0] = 1.0  # 1.65–2.15s
    # 间隔 0.15s > merge_gap 0.1s，不合并；pad 0.2s 后两段本会重叠
    runs = refine_runs(probs, 10.0, merge_gap_s=0.1, pad_s=0.2, min_speech_s=0.3)
    assert len(runs) == 1
    assert runs[0] == pytest.approx((0.8, 2.35))


def test_refine_runs_does_not_merge_across_singing():
    probs = np.zeros((1000, 3), dtype=np.float32)
    probs[100:200, 0] = 1.0
    probs[220:320, 0] = 1.0
    probs[200:220, 1] = 0.9  # 0.2s 间隔是 BGM 歌词
    runs = refine_runs(probs, 10.0, merge_gap_s=0.3, pad_s=0.0, min_speech_s=0.1)
    assert len(runs) == 2
    assert runs[0] == pytest.approx((1.0, 2.0))
    assert runs[1] == pytest.approx((2.2, 3.2))


def test_refine_runs_respects_max_merged():
    probs = np.zeros((1000, 3), dtype=np.float32)
    probs[100:200, 0] = 1.0
    probs[210:310, 0] = 1.0
    runs = refine_runs(probs, 10.0, merge_gap_s=0.3, pad_s=0.0,
                       min_speech_s=0.1, max_merged_s=1.5)
    assert len(runs) == 2
    runs = refine_runs(probs, 10.0, merge_gap_s=0.3, pad_s=0.0,
                       min_speech_s=0.1, max_merged_s=12.0)
    assert len(runs) == 1
    assert runs[0] == pytest.approx((1.0, 3.1))


def test_merge_adjacent_gap_guard_and_max_length():
    segs = [(0.0, 1.0), (1.3, 2.0), (2.5, 3.0)]
    assert merge_adjacent(segs, 0.4, 10.0) == [(0.0, 2.0), (2.5, 3.0)]
    # 间隔守卫拒绝 0.3s 的间隔
    blocked = merge_adjacent(segs, 0.4, 10.0, gap_ok=lambda a, b: b - a < 0.2)
    assert blocked == segs
    # 合并后超过最长限制
    assert merge_adjacent(segs, 0.4, 1.5) == segs
    # 重叠区间直接合并，不询问守卫
    overlap = merge_adjacent([(0.0, 1.0), (0.9, 2.0)], 0.0, 10.0, gap_ok=lambda a, b: False)
    assert overlap == [(0.0, 2.0)]


def test_refine_runs_empty_and_frame_rate():
    assert refine_runs(np.zeros((100, 3), dtype=np.float32), 1.0) == []
    assert refine_runs(np.zeros((0, 3), dtype=np.float32), 1.0) == []
    assert refine_runs(np.zeros((10, 3), dtype=np.float32), 0.0) == []

    probs = np.zeros((500, 3), dtype=np.float32)  # 50fps，不假设 100fps
    probs[100:200, 0] = 1.0
    runs = refine_runs(probs, 10.0, pad_s=0.0, min_speech_s=0.1)
    assert len(runs) == 1
    assert runs[0] == pytest.approx((2.0, 4.0))


def test_filler_detection_and_skip_reason():
    fillers = {_normalize_text(x) for x in ("嗯", "好的", "OK", "对")}
    assert _is_filler("嗯。", fillers)
    assert _is_filler("好好好！", fillers)
    assert _is_filler("ok", fillers)
    assert not _is_filler("谢谢关注，姐姐。", fillers)
    assert not _is_filler("好啊", fillers)

    def row(text, dur=0.6, raw=None):
        return {"asr_text": text, "start_s": 0.0, "end_s": dur, "asr_raw": raw}

    kw = dict(drop_empty=True, drop_fillers=True, filler_max_s=1.2, fillers=fillers)
    assert _skip_reason(row("好的。"), min_conf=None, **kw) == "filler"
    assert _skip_reason(row("好的。", dur=1.5), min_conf=None, **kw) is None
    raw = json.dumps({"text": "谢谢关注", "avg_logprob": -8.0})
    assert _skip_reason(row("谢谢关注", dur=2.0, raw=raw), min_conf=-7.0, **kw) == "low_conf"
    assert _skip_reason(row("谢谢关注", dur=2.0, raw=raw), min_conf=None, **kw) is None
    assert _skip_reason(row("  "), min_conf=None, **kw) == "empty_text"


def test_templates_and_scoring(tmp_path: Path):
    tpl = np.zeros((2, 192), dtype=np.float32)
    tpl[0, 0] = 1.0
    tpl[1, 0] = -1.0
    f = tmp_path / "t.json"
    f.write_text(json.dumps(tpl.tolist()), encoding="utf-8")
    templates = load_templates(f)
    assert np.allclose(np.linalg.norm(templates, axis=1), 1.0)

    embs = np.zeros((2, 192), dtype=np.float32)
    embs[0, 0] = 2.0
    embs[1, 0] = -0.5
    scores, idx = score_embeddings(embs, templates)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0)
    assert idx.tolist() == [0, 1]


def test_template_validation(tmp_path: Path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps([[0.0] * 128]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_templates(f)


def test_avg_logprob_ignores_padding():
    import torch

    from voicedata.adapters.qwen import avg_logprob

    scores = torch.zeros(2, 3, 5)
    seqs = torch.tensor([[1, 2, 3], [1, 2, 4]])
    for b, toks in enumerate(seqs.tolist()):
        for t, tok in enumerate(toks):
            scores[b, t, tok] = 20
    vals = avg_logprob(scores, seqs, pad_id=4)
    assert len(vals) == 2
    assert vals[0] == pytest.approx(0.0, abs=1e-6)
    assert vals[1] is not None and vals[1] > -0.1

    all_pad = avg_logprob(torch.zeros(1, 3, 5), torch.tensor([[4, 4, 4]]), pad_id=4)
    assert all_pad == [None]


def test_config_hashes(make_config):
    cfg = make_config()
    sv = cfg.stage_hash("sv", "job")
    vad = cfg.stage_hash("vad", "job")
    exp40 = cfg.stage_hash("export", "job", 0.4)
    exp50 = cfg.stage_hash("export", "job", 0.5)
    exp55 = cfg.stage_hash("export", "job", 0.55)
    assert sv != vad and exp40 != exp50
    # 改导出阈值不影响声纹阶段；显式给定阈值时导出哈希也不受 sv.threshold 影响
    cfg.apply_overrides({"sv.threshold": 0.55})
    assert cfg.stage_hash("sv", "job") == sv
    assert cfg.stage_hash("export", "job", 0.55) == exp55
    assert cfg.stage_hash("export", "job") == exp55  # 默认阈值跟随配置变为 0.55
    # 改 VAD 参数会失效 VAD 哈希，并且导出必须跟着重算
    cfg.apply_overrides({"vad.speech_threshold": 0.5})
    assert cfg.stage_hash("vad", "job") != vad
    exp_after_vad = cfg.stage_hash("export", "job", 0.55)
    assert exp_after_vad != exp55
    # AED 参数同样属于 VAD 阶段参数
    vad_after = cfg.stage_hash("vad", "job")
    cfg.apply_overrides({"aed.speech_threshold": 0.5})
    assert cfg.stage_hash("vad", "job") != vad_after
    assert cfg.stage_hash("export", "job", 0.55) != exp_after_vad
    # 合并参数和导出过滤参数分别命中 vad / export 哈希
    vad_before_merge = cfg.stage_hash("vad", "job")
    exp_before_merge = cfg.stage_hash("export", "job", 0.55)
    cfg.apply_overrides({"aed.merge_gap_ms": 500, "aed.singing_threshold": 0.3,
                         "aed.max_merged_s": 8.0})
    assert cfg.stage_hash("vad", "job") != vad_before_merge
    assert cfg.stage_hash("export", "job", 0.55) != exp_before_merge
    asr_before = cfg.stage_hash("asr", "job")
    exp_before_filler = cfg.stage_hash("export", "job", 0.55)
    cfg.apply_overrides({"export.min_asr_conf": -7.0, "export.drop_fillers": False})
    assert cfg.stage_hash("asr", "job") == asr_before
    assert cfg.stage_hash("export", "job", 0.55) != exp_before_filler
    # 导出容器/编码属于导出阶段参数，不影响上游
    vad_before_format = cfg.stage_hash("vad", "job")
    exp_before_format = cfg.stage_hash("export", "job", 0.55)
    cfg.apply_overrides({"export.format": "json"})
    assert cfg.stage_hash("vad", "job") == vad_before_format
    assert cfg.stage_hash("export", "job", 0.55) != exp_before_format


def test_db_migrates_old_exports_columns(tmp_path: Path):
    import sqlite3

    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE exports(file_id TEXT PRIMARY KEY, status TEXT, json_path TEXT, "
        "json_sha256 TEXT, threshold REAL, export_params_hash TEXT, segment_count INTEGER, "
        "skipped_count INTEGER, generated_at TEXT, error TEXT)"
    )
    conn.execute(
        "INSERT INTO exports(file_id, status, json_path, json_sha256, segment_count) "
        "VALUES('f', 'done', 'a.json', 'abc123', 1)"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    row = db.one("SELECT * FROM exports WHERE file_id='f'")
    assert row["artifact_path"] == "a.json"
    assert row["artifact_sha256"] == "abc123"
    db.close()


def test_db_recovers_running_rows(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    db.execute(
        "INSERT INTO files(file_id, source_key, rel_path, audio_file, stem) VALUES('f','k','r','a.wav','a')"
    )
    db.execute(
        "INSERT INTO segments(segment_id, file_id, vad_hash, seq, sv_state) VALUES('s','f','v',0,'running')"
    )
    db.execute(
        "INSERT INTO file_stages(file_id, stage, params_hash, status) VALUES('f','vad','v','running')"
    )
    db.commit()
    db.close()

    db2 = Database(tmp_path / "state.db")
    assert db2.get_stage("f", "vad")["status"] == "pending"
    assert db2.one("SELECT sv_state FROM segments WHERE segment_id='s'")["sv_state"] == "pending"
    db2.close()


def test_replace_segments_drops_old_vad_hash(tmp_path: Path):
    db = Database(tmp_path / "state.db")
    db.execute(
        "INSERT INTO files(file_id, source_key, rel_path, audio_file, stem) "
        "VALUES('f','k','r','a.wav','a')"
    )

    def row(sid: str, seq: int) -> dict:
        return {"segment_id": sid, "seq": seq, "start_s": 0.0, "end_s": 1.0,
                "duration_s": 1.0, "wav_path": "x.wav"}

    db.replace_segments("f", "h1", [row("s1", 0), row("s2", 1)])
    db.replace_segments("f", "h2", [row("s3", 0)])
    rows = db.query("SELECT segment_id, vad_hash FROM segments WHERE file_id='f'")
    assert [(r["segment_id"], r["vad_hash"]) for r in rows] == [("s3", "h2")]
    db.close()


def test_upsert_changed_file_resets_decode(make_config):
    cfg = make_config()
    inp = Path(cfg.get("paths.input_dir"))
    inp.mkdir(parents=True)
    p = inp / "x.wav"
    sf.write(str(p), np.zeros(1600, dtype=np.int16), 16000, subtype="PCM_16")
    db = Database(Path(cfg.get("paths.work_root")) / "state.db")
    ids = register(db, cfg)
    fid = ids[0]
    db.set_decode(fid, "done", "h", "w.wav", 0.1)
    assert db.get_file(fid)["decode_state"] == "done"
    p.write_bytes(p.read_bytes() + b"\0" * 100)  # 改大小与 mtime
    register(db, cfg)
    assert db.get_file(fid)["decode_state"] == "pending"
    db.close()


def test_ingest_filters_junk_and_requires_single_excerpt(make_config):
    cfg = make_config()
    inp = Path(cfg.get("paths.input_dir"))
    inp.mkdir(parents=True)
    good = inp / "好.wav"
    sf.write(str(good), np.zeros(32000, dtype=np.int16), 16000, subtype="PCM_16")
    (inp / "._good.wav").write_bytes(b"junk")
    (inp / ".hidden.wav").write_bytes(b"junk")
    (inp / "empty.wav").write_bytes(b"")
    (inp / "notes.txt").write_text("x", encoding="utf-8")
    assert discover(cfg) == [good]

    other = inp / "另一个.wav"
    sf.write(str(other), np.zeros(1600, dtype=np.int16), 16000, subtype="PCM_16")
    db = Database(Path(cfg.get("paths.work_root")) / "state.db")
    with pytest.raises(IngestError):
        register(db, cfg, start_s=0.0, duration_s=1.0)
    ids = register(db, cfg, pattern="好", start_s=0.2, duration_s=0.05)
    row = db.get_file(ids[0])
    assert row["offset_s"] == pytest.approx(0.2)
    assert row["excerpt_duration_s"] == pytest.approx(0.05)
    db.close()


def test_wav_base64_roundtrip(tmp_path: Path):
    import base64

    p = tmp_path / "s.wav"
    sf.write(str(p), np.zeros(320, dtype=np.int16), 16000, subtype="PCM_16")
    raw = base64.b64decode(wav_base64(p))
    assert raw == p.read_bytes()


def test_flac_roundtrip(tmp_path: Path):
    import base64

    from voicedata.audio import flac_info, flac_to_wav_base64, wav_to_flac_bytes

    p = tmp_path / "s.wav"
    sf.write(str(p), np.zeros(3200, dtype=np.int16), 16000, subtype="PCM_16")
    raw = wav_to_flac_bytes(p)
    assert raw[:4] == b"fLaC"
    info = flac_info(raw)
    assert (info["channels"], info["sample_rate"], info["subtype"]) == (1, 16000, "PCM_16")
    assert info["frames"] == 3200
    wav = base64.b64decode(flac_to_wav_base64(raw))
    assert wav[:4] == b"RIFF"
