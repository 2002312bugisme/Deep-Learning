"""LSTM 最小 demo：一个 cell 沿时间反复调用，四个门 + 一条 c 通道。

对应课件《LSTM》两页（含第二页的 peephole 扩展）。跑法：
    D:/software/anaconda/python.exe run_lstm_demo.py
"""

from __future__ import annotations

import torch

from .cell import LSTMLayer

HIDDEN = 3                    # hidden_dim，就是课件里 x1/x2/x3 那三个 dimension
SERIES = [3.0, 1.0, 0.0]      # 三个时间步，每步一个标量输入
WIDTH = 86


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(f"  {line}")


def vec(v: torch.Tensor) -> str:
    return "[" + " ".join(f"{x:+.3f}" for x in v.tolist()) + "]"


def sec_notation() -> None:
    banner("0. 三套记法对照：块图 / 链图 / 代码")
    rows = [
        ("z",      "z^f",        "W_f",     "忘记门的线性输入（还没过激活）"),
        ("f(z_f)", "sigma(z^f)", "sigmoid", "0~1，c^{t-1} 留多少"),
        ("z_i",    "z^i",        "W_i",     "输入门的线性输入"),
        ("f(z_i)", "sigma(z^i)", "sigmoid", "0~1，这次写多少"),
        ("z",      "z",          "W_z",     "候选内容（还没过激活）"),
        ("g(z)",   "tanh",       "tanh",    "写什么，-1~1（链图这条线漏画了 g）"),
        ("z_o",    "z^o",        "W_o",     "输出门的线性输入"),
        ("f(z_o)", "sigma(z^o)", "sigmoid", "0~1，露多少给 a"),
        ("c, c'",  "c^{t-1},c^t", "c",      "记忆本体，走加法通道"),
        ("a",      "h^t",        "h_t",     "这步的输出，也是下一步的输入"),
    ]
    print(f"    {'block diagram':<15}{'chain diagram':<15}{'code':<13}meaning")
    print("    " + "-" * 74)
    for a, b, c, d in rows:
        print(f"    {a:<15}{b:<15}{c:<13}{d}")

    note("""
第三张图把两套记法并排了：右边那个 Block 就是左边链图里的 **一个** 框 ——
一个时间步、一个 cell。链图只是把同一个 Block 沿时间贴了 T 份。

一处对不上的地方：链图里 z 那条线直接进了乘法，没画 tanh；块图里明确有 g。
标准写法是 g = tanh，下面的代码按标准写。
""")


def sec_one_cell() -> LSTMLayer:
    banner("1. 只有一个 cell：参数和序列长度无关")
    layer = LSTMLayer(input_dim=1, hidden_dim=HIDDEN)
    total = sum(p.numel() for p in layer.cell.parameters())
    note(f"""
hidden_dim = {HIDDEN}：每一步的 h^t、c^t 都是 {HIDDEN} 维向量。
四个门各一个 ({HIDDEN}, {1 + HIDDEN}) 的权重矩阵 + {HIDDEN} 个 bias：
        4 x ({HIDDEN} x {1 + HIDDEN} + {HIDDEN}) = {total} 个参数

这 {total} 个就是全部。序列 3 步、30 步、300 步，用的都是这 {total} 个。
""")
    for name, p in layer.cell.named_parameters():
        print(f"    {name:<10}{tuple(p.shape)}")
    return layer


def sec_gates(layer: LSTMLayer) -> None:
    banner("2. 跑一遍：四个门 + 两条状态")
    out = layer([[v] for v in SERIES])
    note("""
输入是 3 个标量。换成词向量只要把 input_dim 改大，整个流程一个字都不用动。
""")
    rows = [("f", "z^f  forget", "c^{t-1} 留多少 (sigmoid)"),
            ("i", "z^i  input ", "这次写多少 (sigmoid)"),
            ("z", "z    cand  ", "写什么内容 (tanh)"),
            ("o", "z^o  output", "露多少给 h^t (sigmoid)")]

    for t, x in enumerate(SERIES):
        print()
        print(f"  t={t + 1}   x^t = {x:+.1f}")
        for key, name, why in rows:
            print(f"    {name:<14}{vec(out[key][t]):<26}{why}")
        print(f"    {'c^t  cell':<14}{vec(out['c'][t]):<26}加法通道，记忆本体")
        print(f"    {'h^t  hidden':<14}{vec(out['h'][t]):<26}→ y^t，也 → 下一个时间步")


def sec_causal(layer: LSTMLayer) -> None:
    banner("3. 水平方向是时间：h^t 只看得见 x^1..x^t")
    a = [[3.0], [1.0], [0.0]]
    b = [[3.0], [1.0], [9.0]]
    out_a, out_b = layer(a), layer(b)

    note("两条输入只差最后一个时间步（0 → 9）。看每个 t 的 h^t 差多少：")
    print()
    print(f"    {'t':>2}  {'x^t':>5}  {'max|dh^t|':>12}   结论")
    for t in range(len(a)):
        d = (out_a["h"][t] - out_b["h"][t]).abs().max().item()
        tag = "精确为 0：后面还没发生的事影响不到它" if d == 0.0 else "才出现差异"
        print(f"    {t + 1:>2}  {a[t][0]:>5.1f}  {d:>12.8f}   {tag}")

    note("""
t=1、t=2 的差值精确等于 0，不是巧合：h^1 只由 x^1 决定。
所以水平方向是时间轴 —— 第 t 步必须等第 t-1 步算完，整条链没法并发。
能并行的是另外两个方向：batch 里的不同句子，和同一时刻 hidden_dim 个单元。
""")


def _residual(out: dict[str, torch.Tensor]) -> list[float]:
    """c^t 里"第 1 步写进去的那部分"还剩多少（按范数比）。"""
    c1 = out["c"][0]
    base = c1.norm().item() or 1.0
    prod = torch.ones_like(c1)
    vals: list[float] = []
    for t in range(out["c"].shape[0]):
        if t > 0:
            prod = prod * out["f"][t]
        vals.append((c1 * prod).norm().item() / base)
    return vals


def sec_forget(layer: LSTMLayer) -> None:
    banner("4. 忘记门：c 到底能记多久")
    xs = [[3.0], [1.0]] + [[0.0]] * 6
    normal = _residual(layer(xs))
    pinned = _residual(layer(xs, forced_forget=1.0))

    note("序列 8 步。看第 1 步写进 c 的那部分，到第 t 步还剩百分之多少：")
    print()
    print(f"    {'':<14}" + "".join(f"{t:>7}" for t in range(1, len(xs) + 1)))
    print(f"    {'normal f':<14}" + "".join(f"{v * 100:>6.0f}%" for v in normal))
    print(f"    {'f := 1.0':<14}" + "".join(f"{v * 100:>6.0f}%" for v in pinned))

    note("""
c^t = z^f 逐元素乘 c^{t-1} + ...，每一步都乘一次 z^f。f 钉死 1.0 就一点不丢；
随机初始化下 f ≈ 0.5，一步砍半，8 步之后基本归零。

两个提醒：
  · 真实训练时 f 是学出来的。LSTM 不是天生能记长程，是训练让它学会"该留就留"。
  · c 走加法通道，梯度沿 c 往回传不用穿过一串 tanh 导数连乘 —— 这才是 LSTM
    缓解梯度消失的结构原因，不是"多了几个门"这么简单。
""")


def sec_peephole() -> None:
    banner("5. 第二页那个红圈（peephole）")
    note("""
它是让三个门直接偷看 c^{t-1}：

    z^f = σ(W_f [x^t ; h^{t-1}] + W_cf ⊙ c^{t-1} + b_f)

标准 LSTM 没有这条线，本 demo 也没实现。要加就是三个 (H, H) 的矩阵，
属于可选项 —— 理解主干的 c 通道之后，这条线只是锦上添花。
""")


def main() -> None:
    torch.manual_seed(7)          # 固定随机权重，结果可复现
    sec_notation()
    layer = sec_one_cell()
    sec_gates(layer)
    sec_causal(layer)
    sec_forget(layer)
    sec_peephole()


if __name__ == "__main__":
    main()
