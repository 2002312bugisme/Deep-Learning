"""统一入口：一个目录一个主题，每个主题一个 demo。

    python main.py              列出所有 demo
    python main.py cnn
    python main.py attention
    python main.py all          按顺序全跑
"""

from __future__ import annotations

import importlib
import sys

DEMOS = ["cnn", "rnn", "lstm", "attention", "transformer", "efficient_attention"]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("用法: python main.py [demo | all]\n")
        for name in DEMOS:
            print(f"  {name}")
        return

    targets = DEMOS if args[0] == "all" else args
    for name in targets:
        if name not in DEMOS:
            print(f"没有这个 demo: {name}（可选: {', '.join(DEMOS)}）")
            continue
        importlib.import_module(f"{name}.demo").main()


if __name__ == "__main__":
    main()
