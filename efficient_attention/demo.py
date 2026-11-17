"""高效 self-attention：为什么 O(N²) 贵，以及五条省算力的路。

跑法（在 D:\\Code\\DeepLearning 下）：
    D:/software/anaconda/python.exe main.py efficient_attention
"""

from __future__ import annotations

import math
import time

import torch

from . import model as M

WIDTH = 84


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(f"  {line}")


def _time(fn, reps: int = 5) -> float:
    fn()                                    # 预热
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def sec_why() -> None:
    banner("1. 问题在哪：N×N 的注意力矩阵")
    d, dv = 64, 64
    print(f"    {'N':>7}{'注意力矩阵':>14}{'元素个数':>16}{'FLOPs (d=d\'=64)':>22}")
    print("    " + "-" * 60)
    for n in (256, 1024, 4096, 16384):
        print(f"    {n:>7}{f'{n}x{n}':>14}{n * n:>16,}{M.flops_full(n, d, dv):>22,.0f}")

    note("""
注意力矩阵是 N×N，算力随序列长度**平方**长。

一张 256×256 的图按像素当 token，N = 65,536，N² 就是 43 亿个格子 —— 放不下。
这就是课件那句「N 大时 self-attention 主导计算」的实际含义。

后面五节是五条路，思路不同，但都在回答同一个问题：那些格子能不能少算一些？
""")


def sec_reorder() -> None:
    banner("2. 忽略 softmax，换乘法顺序：严格等价，算力差几十倍")
    note("""
先看课件那个纯代数的例子（k=1, m=1000, n=1, p=1000）：

    A(CP)  : 先算 CP（m·n·p = 10^6），再乘 A（k·m·p = 10^3）  →  1,001,000
    (AC)P  : 先算 AC（k·m·n = 10^3），再乘 P（k·n·p = 10^3）  →      2,000

结果一样，算力差 500 倍。attention 里是同一招：
""")
    n, d, dv = 2048, 64, 64
    q, k, v = torch.randn(n, d), torch.randn(n, d), torch.randn(n, dv)

    o1 = M.dot_attention(q, k, v)
    o2 = M.linear(q, k, v)
    abs_err = (o1 - o2).abs().max().item()
    rel_err = abs_err / o1.abs().max().item()
    print(f"    数值验证 (N={n})：max|Δ| = {abs_err:.2e}，相对误差 {rel_err:.1e}"
          f"  ← 数学上相等")

    t1 = _time(lambda: M.dot_attention(q, k, v))
    t2 = _time(lambda: M.linear(q, k, v))
    f1, f2 = M.flops_full(n, d, dv), M.flops_linear(n, d, dv)
    print()
    print(f"    {'写法':<24}{'FLOPs':>16}{'耗时':>12}")
    print("    " + "-" * 52)
    print(f"    {'(QKᵀ)V   先算 N×N':<24}{f1:>16,.0f}{t1:>10.2f} ms")
    print(f"    {'Q(KᵀV)   先算 d×d':<24}{f2:>16,.0f}{t2:>10.2f} ms")
    print(f"    {'倍数':<24}{f1 / f2:>15.1f}x{t1 / t2:>10.1f}x")

    note("""
关键：(KᵀV) 是 d×d' 的矩阵，**和 N 完全无关**。N=2048 时算力少 32 倍，N 越大差得越多
（O(N²) vs O(N)）。

代价是：softmax 没了。上面这个等价只在「没有 softmax」时严格成立。
""")


def sec_linear_softmax() -> None:
    banner("3. 想保留 softmax：只能近似（Linear Transformer）")
    n, d, dv = 512, 32, 32
    q, k, v = torch.randn(n, d), torch.randn(n, d), torch.randn(n, dv)

    o_ref, a = M.full(q, k, v)
    o_lin = M.linear_softmax(q, k, v)
    err = (o_ref - o_lin).abs().mean().item() / o_ref.abs().mean().item()
    print(f"    N={n}, d={d}：softmax 版 vs φ 近似版，平均相对误差 {err:.1%}")
    print(f"    注意力矩阵：满的 {tuple(a.shape)}，近似版**从不显式构造**")

    note(f"""
做法：把 exp(q·k) 拆成 φ(q)·φ(k)（φ(x) = elu(x)+1），于是

    b_i = Σ_j φ(q_i)·φ(k_j) v_j / Σ_j φ(q_i)·φ(k_j)
        = [ φ(q_i)ᵀ (Σ_j φ(k_j) v_jᵀ) ] / [ φ(q_i)·Σ_j φ(k_j) ]

分子分母里那两个 Σ 都只跟 j 有关，**算一次给所有 query 复用**。复杂度 O(N²) → O(N)。

但误差 {err:.0%} 一点都不小 —— 因为 exp 是强非线性，elu+1 只在 0 附近像它。
这正是这类方法（Efficient Attention / Linear Transformer / Performer）的真相：
它们不是「换个模块」，而是**必须从头训练**，让网络适应这个近似。
论文里说的「效果不掉」，前提都是同算力重训。
""")


def sec_sparse() -> None:
    banner("4. 稀疏化：局部窗口 / 聚类")
    n, d, dv = 1024, 64, 64
    q, k, v = torch.randn(n, d), torch.randn(n, d), torch.randn(n, dv)

    print(f"    {'做法':<20}{'置 0 比例':>12}{'每行非零':>12}{'耗时':>12}")
    print("    " + "-" * 58)
    print(f"    {'full':<20}{'—':>12}{n:>12}{_time(lambda: M.full(q, k, v)):>10.2f} ms")
    for w in (16, 64, 256):
        _, keep = M.local(q, k, v, w)
        t = _time(lambda: M.local(q, k, v, w))
        print(f"    {f'local w={w}':<20}{M.sparsity(keep):>11.1%}"
              f"{2 * w + 1:>12}{t:>10.2f} ms")
    for c in (4, 16):
        _, keep = M.clustered(q, k, v, c)
        t = _time(lambda: M.clustered(q, k, v, c))
        print(f"    {f'cluster k={c}':<20}{M.sparsity(keep):>11.1%}{'变动':>12}{t:>10.2f} ms")

    note("""
**这张表里最重要的信息是：置 0 了 87% 的格子，耗时反而更高。**

因为这三种写法都是「先算满再置 0」—— 格子省了，计算一步没省，还多了构造 mask、
masked_fill 的开销。聚类的开销更大（k-means 比 attention 本身还贵）。

这就是稀疏 attention 落地难的真实原因：**mask ≠ 加速**，必须配套专门的稀疏 kernel
（Longformer 那种 sliding window 要在 CUDA 里实现）。用 PyTorch 高层 API 写的
「稀疏 attention」在 GPU 上通常比 full attention 还慢。

两条路的思路本身不同：
  · 局部窗口：靠**人类先验**（远处的 token 大概无关），可解释、无额外参数
  · 聚类（Reformer / Routing Transformer）：按相似度分组，但要付聚类代价，
    而且分组是近似的 —— 该看的 pair 可能被分到不同簇
""")


def sec_lowrank() -> None:
    banner("5. 低秩：Linformer 的前提到底成不成立")
    n, d = 256, 32
    torch.manual_seed(0)
    q = torch.randn(n, d)
    k_rand = torch.randn(n, d)
    a_rand = torch.softmax(q @ k_rand.T / math.sqrt(d), -1)

    c, w = torch.randn(n, 4), torch.randn(4, d)      # k 只落在 4 维子空间里
    a_sub = torch.softmax(q @ (c @ w).T / math.sqrt(d), -1)

    print("    A 的归一化奇异值谱（前 8 个）：")
    print(f"      随机 k        {[round(x, 3) for x in M.spectrum(a_rand)]}")
    print(f"      k 落在 4 维子空间 {[round(x, 3) for x in M.spectrum(a_sub)]}")
    print()
    print(f"    {'保留 K 个奇异值':<16}{'随机 k 的误差':>16}{'子空间 k 的误差':>18}")
    print("    " + "-" * 52)
    for kk in (4, 8, 32, 128):
        print(f"    {kk:<16}{M.reconstruct_error(a_rand, kk):>15.1%}"
              f"{M.reconstruct_error(a_sub, kk):>18.1%}")

    note("""
课件的说法是「很多 column 其实是别的 column 的线性组合」，所以可以只留 K 个代表，
把 N×N 压成 N×K。

但上面第一行就是反证：**随机权重下这个谱几乎是平的**，K=32 还差 42%。
就算让 k 落在一个 4 维子空间里（第二列），softmax 一过也不再低秩 ——
K=4 差 71%，要到 K=32 才降到 13%。

结论：**低秩是训练出来的经验性质，不是 softmax 的数学推论。**
Linformer 论文那张谱图也是从训练好的模型里抽的，用随机权重验证不了它。

成本上是真的省：投影+打分合计 2NK(d+d')，N=4096、K=32、d=d'=64 时是 3.4e7，
比 full 的 2.1e9 小 60 倍。代价是 E 本身有 N×K 个参数，N 很大时这笔开销也不小。
""")


def sec_summary() -> None:
    banner("6. 汇总：复杂度与取舍")
    n, d, dv = 4096, 64, 64
    rows = [
        ("full attention", M.flops_full(n, d, dv), "O(N²)", "基线，效果最稳"),
        ("linear（换顺序）", M.flops_linear(n, d, dv), "O(N)", "只有去掉 softmax 才严格等价"),
        ("linear + φ 近似", M.flops_linear(n, d, dv), "O(N)", "必须重训，近似误差大"),
        ("local w=64", M.flops_local(n, d, dv, 64), "O(N·w)", "mask≠加速，要稀疏 kernel"),
        ("low-rank K=32", M.flops_lowrank(n, d, dv, 32), "O(N·K)", "低秩是训练出来的性质"),
    ]
    print(f"    {'方法':<18}{'FLOPs (N=4096)':>18}{'复杂度':>10}   备注")
    print("    " + "-" * 78)
    for name, f, cplx, memo in rows:
        print(f"    {name:<18}{f:>18,.0f}{cplx:>10}   {memo}")

    note("""
五条路的分工：

  · 少算格子：local / stride / global（人类先验）、聚类、Sinkhorn（可学习 pattern）
  · 不构造矩阵：换乘法顺序（Linear Transformer / Performer）
  · 降维度：Linformer（低秩）
  · 干脆不要注意力：Synthesizer（A 当参数学）、MLP-Mixer / FNet

「小孩子才做选择」是这节的精髓：这些 pattern 不互斥，一个 head 用局部、
一个用全局、一个用跨步，混着来更划算（Longformer 就是 sliding + global）。

但要记住两件事：
  1) 论文里「不掉点」基本都是**同算力重训**的结果，不是换个模块就能用；
  2) 先用 profiler 确认 N² 真的是瓶颈，再看数据量撑不撑得住重训。
     很多场景（N<512）full attention 根本不是瓶颈，优化它是白费力气。
""")


def main() -> None:
    torch.manual_seed(0)
    sec_why()
    sec_reorder()
    sec_linear_softmax()
    sec_sparse()
    sec_lowrank()
    sec_summary()


if __name__ == "__main__":
    main()
