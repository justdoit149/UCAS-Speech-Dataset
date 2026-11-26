"""批量推理的降级重试：整批失败则对半拆，单条失败再重试。"""

from __future__ import annotations

import logging
import time
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


def process_batch(
    rows: Sequence[T],
    fn: Callable[[Sequence[T]], Sequence[R]],
    max_attempts: int = 3,
    backoff_s: float = 1.0,
) -> tuple[list[tuple[T, R]], list[tuple[T, str]]]:
    def attempt(items: Sequence[T]) -> tuple[list[tuple[T, R]], list[tuple[T, str]]]:
        last_error = ""
        for i in range(max(1, max_attempts)):
            try:
                out = fn(items)
                if len(out) != len(items):
                    raise RuntimeError(f"批量输出数量不符：{len(out)} != {len(items)}")
                return list(zip(items, out)), []
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if i + 1 < max_attempts:
                    time.sleep(backoff_s * (2**i))
        if len(items) == 1:
            return [], [(items[0], last_error)]
        mid = len(items) // 2
        ok1, fail1 = process_batch(items[:mid], fn, max_attempts, backoff_s)
        ok2, fail2 = process_batch(items[mid:], fn, max_attempts, backoff_s)
        return ok1 + ok2, fail1 + fail2

    return attempt(rows)
