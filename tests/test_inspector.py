from __future__ import annotations

import json
import re
from pathlib import Path

from voicedata.inspector import _collect, build_html


def _row(file: str, start: float, score: float, b64: str = "AAAA") -> dict:
    return {
        "audio_file": file,
        "archive": file + ".zip",
        "start_s": start,
        "end_s": start + 1.0,
        "sv_score": score,
        "noise_prob": 0.5,
        "noisy": 0,
        "asr_text": f"t{start}",
        "b64": b64,
    }


def _write_index(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_collect_samples_near_threshold_but_orders_by_time(tmp_path):
    _write_index(tmp_path, [
        _row("a.mp3", 50.0, 0.39),
        _row("a.mp3", 10.0, 0.90),
        _row("a.mp3", 30.0, 0.41),
    ])

    items = _collect(tmp_path, sample=2, threshold=0.4, pattern=None)
    # 抽样仍选最接近阈值的 0.39 和 0.41，但展示按时间顺序
    assert [item["start"] for item in items] == [30.0, 50.0]


def test_collect_groups_files_and_sorts_each_timeline(tmp_path):
    _write_index(tmp_path, [
        _row("a.mp3", 100.0, 0.8),
        _row("a.mp3", 20.0, 0.7),
        _row("b.mp3", 5.0, 0.9),
        _row("b.mp3", 1.0, 0.6),
    ])

    items = _collect(tmp_path, sample=0, threshold=0.4, pattern=None)
    assert [(item["file"], item["start"]) for item in items] == [
        ("a.mp3", 20.0),
        ("a.mp3", 100.0),
        ("b.mp3", 1.0),
        ("b.mp3", 5.0),
    ]


def test_build_html_embeds_cards_in_time_order(tmp_path):
    out_dir = tmp_path / "out"
    _write_index(out_dir, [
        _row("a.mp3", 50.0, 0.39),
        _row("a.mp3", 10.0, 0.90),
        _row("a.mp3", 30.0, 0.41),
    ])
    page = tmp_path / "calibration.html"

    count = build_html(out_dir, page, sample=2, threshold=0.4)
    assert count == 2

    text = page.read_text(encoding="utf-8")
    match = re.search(r"const T = (\[.*?\]);", text, re.S)
    assert match
    data = json.loads(match.group(1))
    assert [item["start"] for item in data] == [30.0, 50.0]
    assert [item["id"] for item in data] == [0, 1]
