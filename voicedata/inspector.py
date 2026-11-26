"""生成人工抽听校准页（zip/FLAC 与旧 JSON 都支持）。"""

from __future__ import annotations

import html
import json
import logging
import zipfile
from pathlib import Path

from .audio import flac_to_wav_base64

logger = logging.getLogger(__name__)

_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>数据集校准 · {title}</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 24px; background:#fafafa; color:#222; }}
 h1 {{ font-size: 20px; }}
 .bar {{ position: sticky; top: 0; background: #fff; padding: 10px 12px; border: 1px solid #ddd;
        border-radius: 8px; z-index: 10; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }}
 .bar input[type=range] {{ width: 280px; }}
 .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:8px; padding:10px 12px; margin:10px 0; }}
 .meta {{ color:#666; font-size: 13px; margin-bottom:6px; }}
 .text {{ font-size: 15px; margin: 6px 0; }}
 audio {{ width: 100%; max-width: 640px; height: 32px; }}
 button {{ border:1px solid #ccc; background:#fff; border-radius:6px; padding:4px 10px; cursor:pointer; }}
 button.on {{ background:#dff0d8; border-color:#7cb342; }}
 button.no {{ background:#fde0dc; border-color:#e57373; }}
 .ratio {{ font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>数据集校准 · {title}</h1>
<div class="bar">
  <label>阈值 <input id="th" type="range" min="-0.10" max="0.95" step="0.01" value="{threshold}">
    <span id="thv" class="ratio">{threshold:.2f}</span></label>
  <span id="count" class="ratio"></span>
  <button id="csv">导出标注 CSV</button>
  <span class="meta">共 {n} 段（按时间顺序展示；抽样优先选与阈值接近的片段）；标注仅保存在当前页面</span>
</div>
<div id="list">
{cards}
</div>
<script>
const T = {data};
const th = document.getElementById('th');
const thv = document.getElementById('thv');
const countEl = document.getElementById('count');
function refresh() {{
  const t = parseFloat(th.value);
  thv.textContent = t.toFixed(2);
  let kept = 0;
  for (const d of T) {{
    const keep = d.score >= t;
    if (keep) kept++;
    document.getElementById('card-' + d.id).style.opacity = keep ? '1' : '0.45';
  }}
  countEl.textContent = `保留 ${{kept}} / ${{T.length}}`;
}}
th.addEventListener('input', refresh);
function label(id, v, btn) {{
  const d = T.find(x => x.id === id);
  d.label = v;
  for (const b of btn.parentNode.querySelectorAll('button')) b.className = '';
  btn.className = v === 'person' ? 'on' : 'no';
}}
function csv() {{
  const rows = ['file,archive,start_s,end_s,sv_score,noise_prob,asr_conf,asr_text,label'];
  for (const d of T) {{
    const t = (d.text || '').replace(/"/g, '""');
    const c = d.conf === null || d.conf === undefined ? '' : d.conf.toFixed(3);
    rows.push(`"${{d.file}}","${{d.archive}}",${{d.start.toFixed(3)}},${{d.end.toFixed(3)}},${{d.score.toFixed(4)}},${{d.prob.toFixed(4)}},${{c}},"${{t}}",${{d.label || ''}}`);
  }}
  const blob = new Blob([rows.join('\\n')], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels.csv';
  a.click();
}}
document.getElementById('csv').addEventListener('click', csv);
refresh();
</script>
</body>
</html>
"""

_CARD = """<div class="card" id="card-{id}">
 <div class="meta">#{id} · {file} · {arch} · {start:.2f}–{end:.2f}s（{dur:.2f}s） · sv={score:.3f} · conf={conf} · noisy={flag}（p={prob:.3f}）</div>
 <audio controls preload="none" src="data:audio/wav;base64,{b64}"></audio>
 <div class="text">{text}</div>
 <div><button onclick="label({id},'person',this)">本人</button>
      <button onclick="label({id},'other',this)">非本人/无用</button>
      <button onclick="label({id},'unsure',this)">不确定</button></div>
</div>"""


def _normalize_item(d: dict) -> dict:
    out = dict(d)
    out["file"] = str(out.get("audio_file", ""))
    out["archive"] = str(out.get("archive", ""))
    out["audio"] = str(out.get("audio", ""))
    out["start"] = float(out["start_s"])
    out["end"] = float(out["end_s"])
    out["score"] = float(out.get("sv_score") or 0.0)
    out["prob"] = float(out.get("noise_prob") or 0.0)
    out["flag"] = int(out.get("noisy") or 0)
    out["conf"] = out.get("asr_conf")
    out["text"] = str(out.get("asr_text", ""))
    return out


def _read_index(out_dir: Path) -> list[dict] | None:
    path = out_dir / "index.jsonl"
    if not path.exists():
        return None
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(_normalize_item(json.loads(line)))
    except Exception as exc:
        logger.warning("读取 %s 失败：%s", path, exc)
        return None
    return items


def _scan_zips(out_dir: Path) -> list[dict]:
    items = []
    for zpath in sorted(out_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                body = zf.read("segments.jsonl").decode("utf-8")
        except Exception as exc:
            logger.warning("跳过无法读取的 zip：%s：%s", zpath.name, exc)
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["archive"] = zpath.name
            d["audio_file"] = manifest.get("audio_file", zpath.stem)
            items.append(_normalize_item(d))
    return items


def _scan_json(out_dir: Path) -> list[dict]:
    items = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("跳过无法读取的 JSON：%s：%s", path, exc)
            continue
        for seg in doc.get("segments", []):
            d = dict(seg)
            d["archive"] = ""
            d["audio_file"] = doc.get("audio_file", path.stem)
            item = _normalize_item(d)
            item["b64"] = seg.get("audio_base64", "")
            items.append(item)
    return items


def _collect(out_dir: Path, sample: int, threshold: float, pattern: str | None) -> list[dict]:
    items = _read_index(out_dir)
    if not items:
        items = _scan_zips(out_dir)
    if not items:
        items = _scan_json(out_dir)
    if pattern:
        needle = pattern.lower()
        items = [d for d in items if needle in d["file"].lower()
                 or needle in d["archive"].lower()]
    # 抽样仍优先挑与阈值接近的片段，但展示顺序按原音频的时间先后
    file_rank = {}
    for d in items:
        file_rank.setdefault(d["file"], len(file_rank))
    items.sort(key=lambda d: abs(d["score"] - threshold))
    selected = items[:sample] if sample and sample > 0 else items
    selected.sort(key=lambda d: (
        file_rank.get(d["file"], 0), d["start"], d["end"], d["archive"]))
    return selected


def _audio_b64(item: dict, out_dir: Path) -> str | None:
    if item.get("b64"):
        return item["b64"]
    if not item.get("archive") or not item.get("audio"):
        return None
    try:
        with zipfile.ZipFile(out_dir / item["archive"]) as zf:
            raw = zf.read(item["audio"])
        return flac_to_wav_base64(raw)
    except Exception as exc:
        logger.warning("读取片段音频失败：%s/%s：%s", item.get("archive"), item.get("audio"), exc)
        return None


def build_html(
    out_dir: Path,
    out: Path,
    sample: int = 80,
    threshold: float | None = None,
    pattern: str | None = None,
) -> int:
    if threshold is None:
        threshold = 0.4
    items = _collect(out_dir, sample, threshold, pattern)
    cards = []
    data = []
    for item in items:
        b64 = _audio_b64(item, out_dir)
        if b64 is None:
            continue
        idx = len(cards)
        cards.append(_CARD.format(
            id=idx, file=html.escape(item["file"]), arch=html.escape(item["archive"] or "—"),
            start=item["start"], end=item["end"], dur=item["end"] - item["start"],
            score=item["score"], prob=item["prob"], flag=item["flag"],
            conf="—" if item["conf"] is None else f"{float(item['conf']):.2f}",
            b64=b64, text=html.escape(item["text"]),
        ))
        data.append({
            "id": idx, "file": item["file"], "archive": item["archive"],
            "start": item["start"], "end": item["end"],
            "score": item["score"], "prob": item["prob"], "conf": item["conf"],
            "text": item["text"], "label": "",
        })

    archives = sorted({d["archive"] for d in data if d["archive"]})
    if len(archives) == 1:
        title = Path(archives[0]).stem
    elif archives:
        title = f"{len(archives)} 个录音"
    else:
        title = out_dir.name
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(_PAGE.format(
        title=html.escape(title), threshold=threshold, n=len(cards),
        cards="\n".join(cards),
        data=json.dumps(data, ensure_ascii=False),
    ), encoding="utf-8")
    tmp.replace(out)
    return len(cards)
