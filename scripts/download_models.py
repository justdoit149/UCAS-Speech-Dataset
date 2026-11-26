"""幂等下载流水线所需模型（也可以不手动运行，build 会自动检查并下载）。

用法：
    python scripts/download_models.py              # 服务器默认：VAD/SV/1.7B ASR/噪音
    python scripts/download_models.py --only asr06b  # 本地 0.6B ASR
    python scripts/download_models.py --only vad sv noise
    python scripts/download_models.py --list
国内网络：ModelScope 直连；HuggingFace 走 hf-mirror.com（可用 --hf-endpoint 覆盖）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voicedata.models import MODEL_SPECS, download_spec  # noqa: E402

MODELS = ROOT / "models"


def check(key: str, dest: Path) -> bool:
    spec = MODEL_SPECS[key]
    missing = [name for name in spec.files if not (dest / name).exists()]
    if missing:
        print(f"  ! {key} 缺少：{', '.join(missing)}")
        return False
    print(f"  ok {key}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(MODEL_SPECS), default=None)
    ap.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for key, spec in MODEL_SPECS.items():
            print(f"{key:<8} {spec.backend:<12} {spec.repo}  -> models/{spec.subdir}")
        return 0

    keys = args.only or ["vad", "sv", "asr17b", "noise"]
    failed = []
    for key in keys:
        spec = MODEL_SPECS[key]
        dest = MODELS / spec.subdir
        print(f"[{key}] {spec.repo} -> {dest}")
        if check(key, dest):
            continue
        try:
            download_spec(spec, dest, args.hf_endpoint)
        except Exception as exc:
            print(f"  ! 下载失败：{exc}")
            failed.append(key)
            continue
        if not check(key, dest):
            failed.append(key)

    if failed:
        print(f"失败：{', '.join(failed)}")
        return 1
    print("模型已齐全。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
