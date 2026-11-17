"""把中间结果和两句话的差距打印成表格（纯展示，不含任何模型逻辑）。"""

from __future__ import annotations

import torch

from .vocab import Vocab

WIDTH = 86


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip().splitlines():
        print(f"  {line}")


def _vec(v: torch.Tensor) -> str:
    return "[" + " ".join(f"{x:+.3f}" for x in v.tolist()) + "]"


def _top(p: torch.Tensor, vocab: Vocab, k: int = 2) -> str:
    idx = torch.topk(p, k).indices.tolist()
    return "  ".join(f"{vocab.itos[i]} {p[i].item():.3f}" for i in idx)


def _gap(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """返回 (L2 距离, 余弦相似度)。"""
    d = (a - b).norm().item()
    c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return d, c


def show_uni(tokens: list[str], out: dict[str, torch.Tensor], vocab: Vocab, title: str) -> None:
    """单向：一张表看完每步存下来的 a^t。"""
    print()
    print(f"[{title}] 每步的中间结果 a^t 与输出 y^t")
    print(f"{'t':>2}  {'token':<9}{'a^t (stored state)':<32}{'|a^t|':>7}   {'top-2 output'}")
    print("-" * WIDTH)
    for t, (tok, a, p) in enumerate(zip(tokens, out["a"], out["p"]), start=1):
        print(f"{t:>2}  {tok:<9}{_vec(a):<32}{a.norm():>7.3f}   {_top(p, vocab)}")


def show_bi(tokens: list[str], out: dict[str, torch.Tensor], vocab: Vocab, title: str) -> None:
    """双向：两条通道各有一串中间结果，输出由两串拼接而来。"""
    print()
    print(f"[{title}] 正向 / 反向两条通道各自存下来的中间结果")
    print(f"{'t':>2}  {'token':<9}{'a_fwd^t  (reads x^1..x^t)':<32}{'a_bwd^t  (reads x^t..x^T)':<31}")
    print("-" * WIDTH)
    for t, (tok, af, ab) in enumerate(zip(tokens, out["a_fwd"], out["a_bwd"]), start=1):
        print(f"{t:>2}  {tok:<9}{_vec(af):<32}{_vec(ab)}")

    print()
    print(f"[{title}] 融合后的输出 y^t = W_o([a_fwd^t ; a_bwd^t])")
    print(f"{'t':>2}  {'token':<9}{'top-2 output'}")
    print("-" * WIDTH)
    for t, (tok, p) in enumerate(zip(tokens, out["p"]), start=1):
        print(f"{t:>2}  {tok:<9}{_top(p, vocab)}")


def compare(tokens_a: list[str], tokens_b: list[str],
            out_a: dict[str, torch.Tensor], out_b: dict[str, torch.Tensor],
            title: str) -> tuple[float, float]:
    """同一套权重、只换第一个词，逐时间步量一下差多少。返回整句表示的 (d, cos)。"""
    has_bwd = "a_bwd" in out_a
    n = min(len(tokens_a), len(tokens_b), out_a["p"].shape[0], out_b["p"].shape[0])

    print()
    print(f"[{title}] 逐时间步差距（d = L2 距离，cos = 余弦相似度）")
    head = f"{'t':>2}  {'token':<9}{'d(a_fwd)':>10}{'cos':>8}"
    head += f"{'d(a_bwd)':>10}{'cos':>8}" if has_bwd else ""
    head += f"{'d(y)':>10}"
    print(head)
    print("-" * WIDTH)

    for t in range(n):
        line = f"{t + 1:>2}  {tokens_a[t]:<9}"
        if has_bwd:
            d, c = _gap(out_a["a_fwd"][t], out_b["a_fwd"][t])
            line += f"{d:>10.5f}{c:>8.4f}"
            d, c = _gap(out_a["a_bwd"][t], out_b["a_bwd"][t])
            line += f"{d:>10.5f}{c:>8.4f}"
        else:
            d, c = _gap(out_a["a"][t], out_b["a"][t])
            line += f"{d:>10.5f}{c:>8.4f}"
        d, _ = _gap(out_a["p"][t], out_b["p"][t])
        line += f"{d:>10.5f}"
        print(line)

    # 整句表示：单向取最后一步，双向把两条通道各自做平均池化再拼接
    if has_bwd:
        s_a = torch.cat([out_a["a_fwd"].mean(0), out_a["a_bwd"].mean(0)])
        s_b = torch.cat([out_b["a_fwd"].mean(0), out_b["a_bwd"].mean(0)])
        label = "整句表示 (mean-pool 后拼接)"
    else:
        s_a, s_b = out_a["a"][-1], out_b["a"][-1]
        label = "整句表示 (最后一步 a^T)"

    d, c = _gap(s_a, s_b)
    print("-" * WIDTH)
    print(f"  {label}:  d = {d:.5f}   cos = {c:.4f}")
    return d, c


def summary(rows: list[tuple[str, float, float]]) -> None:
    """最后把小表汇总一下：模型 / 整句差距 / 余弦相似度。"""
    print()
    print(f"{'model':<22}{'d(sent)':>10}{'cos':>9}   note")
    print("-" * WIDTH)
    for name, d, c in rows:
        tag = "第一个词的差异基本保住" if c < 0.9 else "第一个词的差异快丢光了"
        print(f"{name:<22}{d:>10.5f}{c:>9.4f}   {tag}")
    print()
