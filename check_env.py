"""检查本机的 conda / python / pytorch：只看不装，不改任何环境。

用法：python check_env.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

CONDA_DIRS = [
    Path(r"D:\Zzz_soft\miniconda3"),
    Path(r"D:\software\anaconda"),
    Path.home() / "miniconda3",
    Path.home() / "anaconda3",
    Path(r"C:\ProgramData\miniconda3"),
    Path(r"C:\ProgramData\anaconda3"),
]

TORCH_PROBE = (
    "import torch;"
    "print(torch.__version__, torch.version.cuda or 'cpu', torch.cuda.is_available())"
)


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:                       # noqa: BLE001
        return f"ERROR: {exc}"
    text = (r.stdout or r.stderr).strip()
    return text.splitlines()[-1] if text else "(empty)"


def scan_envs() -> list[tuple[str, Path]]:
    """把各 conda 安装下的 base 和 envs/* 都列出来。"""
    found: list[tuple[str, Path]] = []
    for root in CONDA_DIRS:
        if not root.is_dir():
            continue
        base_py = root / "python.exe"
        if base_py.is_file():
            found.append((f"{root.name} (base)", base_py))
        envs_dir = root / "envs"
        if envs_dir.is_dir():
            for d in sorted(envs_dir.iterdir()):
                py = d / "python.exe"
                if py.is_file():
                    found.append((f"{root.name}/{d.name}", py))
    return found


def main() -> None:
    print("=" * 78)
    print("  conda / python / pytorch 体检")
    print("=" * 78)

    # ---- 当前解释器 ----
    print("\n[当前解释器]")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")

    # ---- conda ----
    print("\n[conda]")
    on_path = shutil.which("conda")
    print(f"  conda 在 PATH 里 : {on_path or '否 (需要 conda init 或用绝对路径)'}")
    found = [d for d in CONDA_DIRS if (d / "Scripts" / "conda.exe").is_file()]
    if not found:
        print("  安装目录         : 未找到 ( 检查 D:\\Zzz_soft\\miniconda3 等常见位置 )")
    for d in found:
        kind = "Anaconda" if "anaconda" in d.name.lower() else "Miniconda"
        print(f"  安装目录         : {d}  ({kind})")
        print(f"  conda 版本       : {run([str(d / 'Scripts' / 'conda.exe'), '--version'])}")

    # ---- 各环境 python / torch ----
    print("\n[各环境]")
    print(f"  {'environment':<26}{'python':<10}{'torch':<18}{'cuda'}")
    print("  " + "-" * 74)
    for name, py in scan_envs():
        pyv = run([str(py), "-V"]).replace("Python ", "")
        torch_info = run([str(py), "-c", TORCH_PROBE])
        if "No module named" in torch_info or "ERROR" in torch_info:
            tv, cuda, tag = "-", "-", "缺 torch"
        else:
            parts = torch_info.split()
            tv = parts[0] if parts else "?"
            cuda = parts[1] if len(parts) > 1 else "?"
            tag = "可用" if parts[-1] == "True" else ("CPU" if cuda == "cpu" else f"cu{cuda}")
        print(f"  {name:<26}{pyv:<10}{tv:<18}{tag}")

    print("\n[结论]")
    print("  这个 demo 只用 CPU 就够，挑上面任意一个 torch 可用的环境跑 run_demo.py 即可。")


if __name__ == "__main__":
    main()
