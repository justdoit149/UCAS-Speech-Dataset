"""命令行入口：build / status / inspect / validate。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import STAGES, Config, ConfigError
from .models import ModelError
from .pipeline import Pipeline, PipelineBusy

DEFAULT_CONFIG = "configs/server.yaml"


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"配置文件路径（默认 {DEFAULT_CONFIG}）")
    p.add_argument("--job", default="full", help="任务名，决定 work/<job> 与 out/<job>")
    p.add_argument("--device", default=None, help="覆盖 run.device：cpu / cuda:0 / auto")
    p.add_argument("--dtype", default=None, help="覆盖 run.dtype：float32 / bfloat16 ...")


def _load(args) -> Config:
    cfg = Config.load(args.config)
    cfg.apply_overrides({"run.device": args.device, "run.dtype": args.dtype})
    return cfg


def cmd_build(args) -> int:
    cfg = _load(args)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    try:
        with Pipeline(cfg, args.job) as pipe:
            ids = pipe.run(
                pattern=args.file,
                start_s=args.start,
                duration_s=args.duration,
                stages=stages,
                force=args.force,
                retry_failed=args.retry_failed,
                limit=args.limit,
                threshold=args.threshold,
            )
    except PipelineBusy as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except ModelError as exc:
        print(f"模型准备失败：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断：当前批次已提交，重新运行同一条命令即可续跑。", file=sys.stderr)
        return 130
    print(f"完成：{len(ids)} 个文件，job={args.job}，work={cfg.job_dir('work', args.job)}")
    return 0


def _stage_summary(row) -> str:
    if row is None:
        return "-"
    status = row["status"]
    stats = {}
    try:
        stats = json.loads(row["stats_json"]) if row["stats_json"] else {}
    except Exception:
        pass
    bits = [status]
    if "segment_count" in stats:
        bits.append(f"{stats['segment_count']}段")
    if stats.get("failed"):
        bits.append(f"失败{stats['failed']}")
    if status in ("failed", "skipped") and row["error"]:
        bits.append(row["error"][:40])
    return " ".join(bits)


def cmd_status(args) -> int:
    cfg = _load(args)
    from .db import Database

    work = cfg.job_dir("work", args.job)
    db_path = work / "state.db"
    if not db_path.exists():
        print(f"没有找到状态库：{db_path}", file=sys.stderr)
        return 1
    db = Database(db_path, readonly=True)
    rows = db.files()
    payload = []
    try:
        for row in rows:
            entry = {
                "audio_file": row["audio_file"],
                "file_id": row["file_id"],
                "offset_s": row["offset_s"],
                "excerpt_duration_s": row["excerpt_duration_s"],
                "decode": row["decode_state"],
                "stages": {},
            }
            for stage in STAGES:
                st = db.get_stage(row["file_id"], stage)
                entry["stages"][stage] = {
                    "status": st["status"] if st else None,
                    "error": st["error"] if st else None,
                    "stats": json.loads(st["stats_json"]) if st and st["stats_json"] else None,
                }
            ex = db.get_export(row["file_id"])
            if ex:
                keys = ex.keys()
                entry["export"] = {
                    "status": ex["status"],
                    "path": (ex["artifact_path"] if "artifact_path" in keys
                             else ex["json_path"]),
                    "segments": ex["segment_count"], "threshold": ex["threshold"],
                }
            payload.append(entry)
    finally:
        db.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for e in payload:
        print(f"\n{e['audio_file']}  [{e['file_id']}]  decode={e['decode']}")
        for stage in STAGES:
            if stage == "export":
                continue
            print(f"  {stage:<7} {e['stages'][stage]['status'] or '-'}")
        if "export" in e:
            print(f"  export  {e['export']['status']} -> {e['export']['path']}"
                  f" ({e['export']['segments']} 段, 阈值 {e['export']['threshold']})")
    if not payload:
        print("（还没有注册任何文件）")
    return 0


def cmd_inspect(args) -> int:
    cfg = _load(args)
    from .inspector import build_html

    out_dir = cfg.job_dir("output", args.job)
    if not out_dir.exists():
        print(f"输出目录不存在：{out_dir}", file=sys.stderr)
        return 1
    if not (any(out_dir.glob("*.zip")) or (out_dir / "index.jsonl").exists()
            or any(out_dir.glob("*.json"))):
        print("没有可用的导出产物（zip/index.jsonl/旧 JSON）", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else (out_dir / "calibration.html")
    count = build_html(
        out_dir, out, sample=args.sample, threshold=args.threshold, pattern=args.file)
    print(f"校准页已生成：{out}（{count} 段）")
    return 0


def cmd_validate(args) -> int:
    cfg = _load(args)
    from .validate import validate_job

    report = validate_job(cfg, args.job, file_pattern=args.file)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["files"]:
            flag = "OK" if item["ok"] else "FAIL"
            print(f"[{flag}] {item['artifact']}")
            for msg in item["errors"]:
                print(f"    - {msg}")
            for msg in item["warnings"]:
                print(f"    ! {msg}")
        index = report["index"]
        if index.get("path"):
            flag = "OK" if index["ok"] else "FAIL"
            print(f"[{flag}] 索引 {index['path']}（{index['segments']} 段，"
                  f"{index['file_count']} 个录音）")
            for msg in index["errors"]:
                print(f"    - {msg}")
            for msg in index["warnings"]:
                print(f"    ! {msg}")
        print(f"\n合计：{report['ok_count']}/{report['file_count']} 通过，"
              f"{report['total_segments']} 段，{report['total_speech_s']:.1f}s 语音")
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicedata", description="直播语音数据集流水线")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="运行流水线（可续跑）")
    _add_common(b)
    b.add_argument("--file", default=None, help="只处理文件名包含该子串的文件")
    b.add_argument("--start", type=float, default=None, help="片段起点（秒）")
    b.add_argument("--duration", type=float, default=None, help="片段时长（秒）")
    b.add_argument("--stages", default=None, help=f"逗号分隔的阶段子集：{','.join(STAGES)}")
    b.add_argument("--force", action="store_true", help="强制重跑所选阶段")
    b.add_argument("--retry-failed", action="store_true", help="把 failed/partial 的阶段和段重新入队")
    b.add_argument("--limit", type=int, default=None, help="最多处理前 N 个文件")
    b.add_argument("--threshold", type=float, default=None, help="导出阈值（只影响导出）")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("status", help="查看状态")
    _add_common(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    i = sub.add_parser("inspect", help="生成人工抽听校准页")
    _add_common(i)
    i.add_argument("--file", default=None, help="只包含原音频文件名匹配的段")
    i.add_argument("--out", default=None)
    i.add_argument("--sample", type=int, default=80, help="抽取的段数（按接近阈值的程度）")
    i.add_argument("--threshold", type=float, default=None)
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("validate", help="校验导出 zip / 旧 JSON 与全局索引")
    _add_common(v)
    v.add_argument("--file", default=None, help="只校验文件名匹配的产物")
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
