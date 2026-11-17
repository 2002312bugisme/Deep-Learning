"""Self-attention 最小 demo：Q/K/V -> A -> A' -> O，外加位置编码与并行性。

严格照课件的矩阵形状走（I 是 d_model x N，一个 token 一列）：
    Q = W^q I,  K = W^k I,  V = W^v I          (d_k x N)
    A = K^T Q                                   (N x N), A[j][i] = k^j · q^i
    A' = softmax(A, 每一列归一化)                (N x N)
    O = V A'                                    (d_v x N)

跑法：
    D:/software/anaconda/python.exe run_attention_demo.py
"""

from __future__ import annotations

import time

import torch

from .model import SelfAttention, sinusoidal_pe

WIDTH = 82

# 为了让数字看得懂，用 one-hot 当输入、手设整数权重（课件里 a^1..a^4 就是 one-hot）。
# 真实模型里 W^q / W^k / W^v 全是学出来的 —— 就是课件第 12 张红圈标的那三个矩阵。
W_Q = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]])
W_K = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0]])
W_V = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0]])


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(f"  {line}")


def show(name: str, t: torch.Tensor, fmt: str = "{:+7.3f}") -> None:
    print(f"    {name:<3} ({t.shape[0]}x{t.shape[1]})")
    for row in t.tolist():
        print("        " + " ".join(fmt.format(v) for v in row))
    print()


def sec_flow() -> SelfAttention:
    banner("1. 走一遍：I -> Q,K,V -> A -> A' -> O")
    attn = SelfAttention(W_Q, W_K, W_V)
    x = torch.eye(4)                          # a^1..a^4 是 one-hot，一列一个 token
    out = attn(x)

    note("4 个 token，d_model=4，d_k=d_v=3。I 的每一列就是一个 a^i（和课件一样是 one-hot）。")
    print()
    show("I", out["I"], "{:+7.1f}")
    show("Q", out["Q"])
    show("K", out["K"])
    show("V", out["V"])
    show("A", out["A"], "{:+7.1f}")
    show("A'", out["A'"])
    show("O", out["O"])

    col_sum = out["A'"].sum(dim=0)
    manual = (out["Q"][:, 0] * out["K"][:, 2]).sum()
    note(f"""
手工验算：q^1·k^3 = {manual:.1f}，而 A 的第 1 列第 3 行就是 {out["A"][2][0]:.1f}
—— 对得上，这就是课件里 a_{{1,3}} = k^3·q^1。

A' 每一列加起来 = {tuple(round(v, 3) for v in col_sum.tolist())}
（softmax 只在"同一个 query 对所有 key"这一列内做，所以每列和为 1）
""")
    return attn


def sec_no_position(attn: SelfAttention) -> None:
    banner("2. 没有位置信息：打乱顺序，输出只是跟着打乱")
    x = torch.eye(4)
    perm = torch.tensor([2, 0, 3, 1])

    o1 = attn(x)["O"]
    o2 = attn(x[:, perm])["O"]
    d0 = (o2 - o1[:, perm]).abs().max().item()

    print(f"    原始顺序       O 的第 1 列 = {[round(v, 3) for v in o1[:, 0].tolist()]}")
    print(f"    打乱后顺序     O 的第 1 列 = {[round(v, 3) for v in o2[:, 0].tolist()]}")
    print(f"    打乱后想对上原始的第 3 列：最大差值 {d0:.2e}")
    note("""
差值是 0（浮点误差级别），意味着：打乱输入之后，输出只是按同样的顺序重新排了一下。

换句话说，对 self-attention 来说 (a^1, a^2, a^3) 和 (a^3, a^1, a^2) 是**同一个集合**，
它根本不知道谁在前谁在后。这就是课件那句 "No position information in self-attention"。
""")


def sec_position(attn: SelfAttention) -> None:
    banner("3. 加上位置编码：顺序才开始有区别")
    x = torch.eye(4)
    perm = torch.tensor([2, 0, 3, 1])
    pe = sinusoidal_pe(4, 4)

    show("PE", pe)
    note("上面是 hand-crafted 的正弦/余弦编码：每个位置一个唯一向量，不同维度频率不同。")

    o1 = attn(x, pe)["O"]
    o2 = attn(x[:, perm], pe)["O"]           # 注意：PE 按"位置槽位"固定，只打乱 token
    d1 = (o2 - o1[:, perm]).abs().max().item()
    print(f"    加 PE 后，打乱 token、位置槽位不动：最大差值 {d1:.4f}")

    learned = torch.nn.Embedding(4, 4)
    print(f"    可学习方式：nn.Embedding(4, 4) 就是一张 4x4 的表，"
          f"{learned.weight.numel()} 个参数，"
          f"requires_grad={learned.weight.requires_grad}")
    note(f"""
加了 PE 之后差值是 {d1:.2f}（原来是 0）—— 顺序变了，表示就变了，模型终于能区分"谁在前"。

两条路都行：
  · hand-crafted：正弦/余弦固定不动，不占参数（原论文 Transformer 用的这个）
  · learned：就是一张 N x d 的表，跟词向量一样靠梯度学（BERT 用的这个）
本 demo 两种都能加，因为 PE 只是和输入相加。
""")


def sec_multihead() -> None:
    banner("4. 多头：几套 W^q/W^k/W^v 并行，各看一种关联")
    g = torch.Generator().manual_seed(3)
    heads = [SelfAttention(torch.randn(2, 4, generator=g),
                           torch.randn(2, 4, generator=g),
                           torch.randn(2, 4, generator=g)) for _ in range(2)]
    x = torch.eye(4)
    outs = [h(x) for h in heads]

    show("A'1", outs[0]["A'"])
    show("A'2", outs[1]["A'"])
    corr = torch.corrcoef(torch.stack([outs[0]["A"].flatten(),
                                       outs[1]["A"].flatten()]))[0, 1].item()
    note(f"两个 head 的注意力矩阵不同（相关系数 {corr:.3f}）。"
         "随机权重下它们只是碰巧不同，没有语义；")

    cat = torch.cat([outs[0]["O"], outs[1]["O"]], dim=0)   # (4, 4) = 2 个 (2,4) 拼起来
    w_o = torch.randn(4, 4, generator=g)
    b = w_o @ cat
    note(f"""
真实训练时它们会分化：有的 head 盯语法、有的盯指代、有的盯相邻位置。

拼接再降维：两个 head 各出 2 维 -> concat 成 {tuple(cat.shape)} -> 乘 W^O
-> 回到 d_model=4 维 {tuple(b.shape)}。W^O 也是要学的参数，课件第 14 张那个蓝框。
""")


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def _bench(length: int, d: int = 64, reps: int = 5) -> tuple[float, float]:
    x = torch.randn(d, length)
    w = torch.randn(d, d)
    h0 = torch.zeros(d)

    def run_attn() -> None:
        q = w @ x
        p = torch.softmax(q.T @ q, dim=0)
        _ = q @ p

    def run_rnn() -> None:
        h = h0
        for t in range(length):
            h = torch.tanh(w @ h + x[:, t])

    run_attn(), run_rnn()                       # 预热，别把首次调用的开销算进去
    ta = min(_timeit(run_attn) for _ in range(reps))
    tr = min(_timeit(run_rnn) for _ in range(reps))
    return ta, tr


def sec_speed() -> None:
    banner("5. 和 RNN 的速度对比：并行 vs 串行")
    print(f"    {'L':<6}{'self-attention':>16}{'逐时间步(类RNN)':>18}{'倍数':>9}")
    print("    " + "-" * 52)
    rows = []
    for length in (64, 256, 1024, 2048):
        ta, tr = _bench(length)
        rows.append((length, ta, tr))
        print(f"    {length:<6}{ta:>13.2f} ms{tr:>15.2f} ms{tr / ta:>8.1f}x")

    best = max(rows, key=lambda r: r[2] / r[1])
    note(f"""
结构上的差别只有一句话：attention 一层之内所有位置同时算完（1 个串行依赖），
RNN 必须等第 t-1 步（串行 L 步，课件标 nonparallel 的地方）。

但倍数不是随长度一路涨：L={best[0]} 时 {best[2] / best[1]:.1f} 倍最大，再长反而回落 ——
因为 attention 是 O(L^2)，L=2048 时注意力矩阵就有 400 万个元素。这正好对上课件后半段：
attention 靠并行赢 RNN，代价是长度平方，序列再长就要 Truncated Self-attention。

两点必须说清楚，不然这个对比会骗人：
  1) 这里的 RNN 是 Python 逐时间步循环，cuDNN 融合实现会快很多，倍数会缩小，
     但"串行 L 步"这个结构限制改不了。
  2) 几十个 token 的短序列上 attention 未必更快。它真正赢的是：并行度对 GPU 友好，
     以及任意两个位置之间只隔一步 —— RNN 里那句话要跨 L 步才能传过去。
""")


def main() -> None:
    torch.manual_seed(0)
    attn = sec_flow()
    sec_no_position(attn)
    sec_position(attn)
    sec_multihead()
    sec_speed()


if __name__ == "__main__":
    main()
