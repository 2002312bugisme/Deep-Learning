"""Transformer 的 block：FFN 干什么、BN 为什么不如 LN、pre/post-norm、encoder-decoder。

跑法（在 D:\\Code\\DeepLearning 下）：
    D:/software/anaconda/python.exe main.py transformer
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import DecoderBlock, EncoderBlock, FFN, MultiHeadAttention
from .norm import BatchNormSeq, LayerNorm

WIDTH = 84


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(f"  {line}")


def sec_ffn() -> None:
    banner("1. Feed Forward 是什么，干什么")
    d, d_ff = 512, 2048
    ffn = FFN(d, d_ff)
    attn = MultiHeadAttention(d, 8)
    x = torch.randn(2, 5, d)

    p_ffn = sum(p.numel() for p in ffn.parameters())
    p_attn = sum(p.numel() for p in attn.parameters())
    print(f"    输入 {tuple(x.shape)}  ->  {tuple(ffn(x).shape)}")
    print(f"    d_model={d}, d_ff={d_ff}：FFN 参数 {p_ffn:,}，attention 参数 {p_attn:,}"
          f"，FFN 占 {p_ffn / (p_ffn + p_attn):.0%}")

    # 改一个 token，看哪些位置的输出跟着变
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32)
    x2 = x.clone()
    x2[0, 1] += 5.0                                   # 只动第 2 个 token
    f, a = FFN(32), MultiHeadAttention(32, 4)
    with torch.no_grad():
        d_ffn = (f(x2) - f(x)).abs().amax(dim=-1)[0]
        d_attn = (a(x2) - a(x)).abs().amax(dim=-1)[0]

    print()
    print(f"    {'位置':<6}{'过 FFN 的变化':>16}{'过 attention 的变化':>22}")
    for i in range(4):
        print(f"    {i + 1:<7}{d_ffn[i]:>14.4f}{d_attn[i]:>20.4f}")

    note("""
形状上就是 d_model -> 4*d_model -> d_model，中间夹一个 ReLU。
一个 block 里 2/3 的参数在 FFN 上，不在 attention 上 —— 这点很容易搞反。

看上面那张表：改动第 2 个 token 之后，
  · FFN 这一列只有第 2 位变化，其余**精确为 0** —— 它对每个位置独立作用，位置之间不通信
  · attention 这一列四个位置全变了 —— 因为大家都得 attend 第 2 个 token

所以分工是：attention 负责跨位置"通信"，FFN 负责在每个位置内部"计算"。
attention 的输出是 V 的加权平均（权重非负、和为 1），只能混合已有的东西；
要造新的特征，只能靠 FFN 这个逐位置的非线性映射。
""")


def sec_norm() -> None:
    banner("2. BatchNorm 和 LayerNorm 差在哪：归一化的轴")
    note("""
    x 的形状 (B, L, d) = (batch, 序列长度, d_model)

    LayerNorm  在最后一维 d 上算 mean/std  -> 一个 token 自己归一化，和别人无关
    BatchNorm  在 (B, L) 上算             -> 同一个维度跨所有样本、所有位置归一化

所以这两个不是"差不多的两种写法"，是**统计量取自哪些元素**的根本区别。
""")

    torch.manual_seed(0)
    d, lens = 8, [6, 3, 1]
    n = max(lens)
    x = torch.randn(len(lens), n, d) * 2 + 1
    mask = torch.zeros(len(lens), n, dtype=torch.bool)
    for i, ln_ in enumerate(lens):
        mask[i, :ln_] = True
    x = x * mask.unsqueeze(-1)                        # padding 位置填 0

    bn = BatchNormSeq(d)
    bn_m = BatchNormSeq(d, masked=True)
    ln = LayerNorm(d)
    with torch.no_grad():
        o_bn, o_bnm, o_ln = bn(x), bn_m(x, mask), ln(x)

    solo_bn = bn(x[:1])[0]
    solo_ln = ln(x[:1])[0]
    print(f"    ① padding 污染：样本 1 整条都是真 token（长度 {lens[0]}）")
    print(f"       BN 把 padding 算进统计 vs 只用真 token（masked BN）："
          f"最大差 {(o_bn[0] - o_bnm[0]).abs().max().item():.4f}")
    print(f"       LayerNorm 不需要 mask，本来就不受影响："
          f"最大差 {(o_ln[0] - ln(x, mask)[0]).abs().max().item():.2e}")

    print(f"\n    ② batch 依赖：同一句话，单独跑 vs 拼在 batch 里跑")
    print(f"       LayerNorm  max|Δ| = {(solo_ln - o_ln[0]).abs().max().item():.2e}")
    print(f"       BatchNorm  max|Δ| = {(solo_bn - o_bn[0]).abs().max().item():.4f}")

    note("""
① BN 把 padding 的 0 也算进 mean/std，样本 1 明明全是真 token，输出却被别人拉偏；
   masked BN（PowerNorm 那篇的做法）只统计真 token，差值是 0。

② 同一句话单独跑和拼 batch 跑：LN 完全一致，BN 直接变了个样 ——
   因为 BN 的统计量里掺进了别的样本。

Transformer 用 LN 的两个理由：
  1) 序列长度不齐，padding 会污染 BN 的统计（这条 masked BN 能修掉）
  2) BN 让"一个样本的表示"依赖同 batch 的其他人 —— 训练时 batch 组成一变统计就变，
     推理时（尤其自回归生成、batch=1）根本对不上训练时的分布。LN 每个 token 自己算，
     训练和推理完全一致。

顺带纠正一个说法：不是"BN 本质更差"，是"BN 的前提在变长序列里不成立"。
arXiv 2003.07845 (PowerNorm) 就在讲这个：把 masking 修对之后，BN 在 Transformer 里也能打。
""")


def _stack(pre_norm: bool, depth: int, d: int, heads: int, d_ff: int) -> nn.ModuleList:
    torch.manual_seed(1)
    return nn.ModuleList([EncoderBlock(d, heads, d_ff, pre_norm) for _ in range(depth)])


def _probe(stack: nn.ModuleList, x: torch.Tensor,
           w: torch.Tensor) -> tuple[list[float], list[float]]:
    norms = []
    for blk in stack:
        x = blk(x)
        norms.append(x.norm(dim=-1).mean().item())
    (x * w).sum().backward()                    # 一个固定随机线性读数当 loss
    grads = []
    for blk in stack:
        g = sum(p.grad.pow(2).sum().item() for p in blk.parameters() if p.grad is not None)
        grads.append(g ** 0.5)
    return norms, grads


def sec_prepost() -> None:
    banner("3. post-norm vs pre-norm：为什么原版不是最优")
    depth, d, heads, d_ff = 12, 64, 4, 128
    x0 = torch.randn(4, 16, d)
    w = torch.randn(1, 1, d)                    # 固定的随机读数方向

    o_post, g_post = _probe(_stack(False, depth, d, heads, d_ff), x0.clone(), w)
    o_pre, g_pre = _probe(_stack(True, depth, d, heads, d_ff), x0.clone(), w)

    print(f"    {depth} 层 encoder block，同一份初始化，只改 norm 的位置")
    print()
    print(f"    {'层':<5}{'‖x‖ post':>11}{'‖x‖ pre':>11}{'‖grad‖ post':>15}{'‖grad‖ pre':>14}")
    for i in range(depth):
        if i + 1 in (1, 2, 3, 6, 9, 12):
            print(f"    {i + 1:<6}{o_post[i]:>10.2f}{o_pre[i]:>11.2f}"
                  f"{g_post[i]:>15.2e}{g_pre[i]:>14.2e}")

    note("""
两个写法只差 norm 放在哪里：
    post-norm   x = LN(x + Sublayer(x))    ← 原论文
    pre-norm    x = x + Sublayer(LN(x))    ← 后来的主流（GPT 系基本都是）

这张表能确定的是 ‖x‖：post 被每层的 LN 反复拉回同一尺度，12 层一直卡在 8.00；
pre 的残差流一路涨 —— 因为加法路径上不再有 LN 挡着，信息可以一路累加。

梯度这一栏要老实说：这组随机初始化下两者相当（都是 2~3e3），**单次反传看不出
post-norm 的劣势**。pre-norm 的好处是训练动态上的 —— 早期更稳、深层不易发散、
可以去掉 learning-rate warmup，那必须真训练才复现得了。

arXiv 2002.04745 做的就是真实训练对比，结论是 pre-norm 更稳、能省掉 warmup。
所以"原始架构不是最优"这句成立，但它改的只是 norm 的位置，
self-attention 和 FFN 本身没动 —— 这也是为什么后来大家改架构，改的多是 norm 和激活，
而不是注意力那套机制。
""")


def sec_seq2seq() -> None:
    banner("4. encoder / decoder：Q 从哪来，K/V 从哪来")
    d = 16
    enc = EncoderBlock(d, 4, 32)
    dec = DecoderBlock(d, 4, 32)
    src = torch.randn(2, 5, d)          # batch=2，源句 5 个 token
    tgt = torch.randn(2, 3, d)          # 目标句 3 个 token

    memory = enc(src)
    out = dec(tgt, memory)
    print(f"    src {tuple(src.shape)}  --encoder-->  memory {tuple(memory.shape)}")
    print(f"    tgt {tuple(tgt.shape)} + memory  --decoder-->  {tuple(out.shape)}")
    print()
    print("    三个 attention 的 Q/K/V 来源：")
    print("      decoder self-attn   Q/K/V 全来自 decoder 自己，加因果掩码")
    print(f"        A 形状 {tuple(dec.self_attn.last_attn.shape)}  = (batch, heads, 3, 3)")
    print("      decoder cross-attn  Q 来自 decoder，K/V 来自 encoder")
    print(f"        A 形状 {tuple(dec.cross_attn.last_attn.shape)}  = (batch, heads, 3, 5)")

    print()
    print("    因果掩码长这样（X = 看不到，. = 能看到）：")
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    for row in mask.tolist():
        print("        " + " ".join("X" if v else "." for v in row))

    note("""
这里要纠正一句：不是"encoder 决定要输出什么 sequence"。
encoder 只负责把源句整理成 memory（一串 K/V），decoder 每一步用自己的 Q 去里面查，
再自己决定下一个 token。决定权在 decoder，encoder 只提供"可查的键值对"。

顺带两个细节：
  · 训练时 decoder 的输入是"右移一位"的正确答案（teacher forcing），
    所以一次前向就能并行算完所有位置 —— 因果掩码保证第 t 位看不到第 t+1 位。
  · 推理时没有正确答案可用，只能一个 token 一个 token 生成，这时才是串行的。
""")


def main() -> None:
    torch.manual_seed(0)
    sec_ffn()
    sec_norm()
    sec_prepost()
    sec_seq2seq()


if __name__ == "__main__":
    main()
