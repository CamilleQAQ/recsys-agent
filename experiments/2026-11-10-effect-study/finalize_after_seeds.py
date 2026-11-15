"""等指定种子写进 seed_study.json 后，自动重算聚合、逐阶段表与两张图。

    python finalize_after_seeds.py 46 [bundled_python_with_pillow]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMEOUT_MINUTES = 240


def main() -> int:
    args = [value for value in sys.argv[1:] if not value.startswith("--")]
    target = args[0] if args else "46"
    bundled = args[1] if len(args) > 1 else ""
    path = ROOT / "seed_study.json"
    deadline = time.time() + TIMEOUT_MINUTES * 60
    while time.time() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if target in payload.get("per_seed", {}):
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(30)
    else:
        print(f"超时：seed {target} 仍未写入", flush=True)
        return 1

    print(f"seed {target} 已写入，开始重算", flush=True)
    for script in ("make_aggregate.py", "make_stage_table.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / script)],
            capture_output=True, text=True, encoding="utf-8",
        )
        print(f"== {script} rc={result.returncode}", flush=True)
        print(result.stdout or result.stderr, flush=True)

    figures = ROOT / "make_figures.py"
    if bundled and Path(bundled).exists():
        result = subprocess.run(
            [bundled, str(figures)], capture_output=True, text=True, encoding="utf-8"
        )
        print(f"== make_figures.py rc={result.returncode}", flush=True)
        print(result.stdout or result.stderr, flush=True)
    else:
        print("未提供带 Pillow 的解释器，跳过画图", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
