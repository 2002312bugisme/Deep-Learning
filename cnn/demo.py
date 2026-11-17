"""CNN 最小 demo：感受野 / 参数量 / 通道数 / pooling 不变性。

四个小块，对应课件里容易被讲混的四个点。跑法：
    D:/software/anaconda/python.exe run_cnn_demo.py
"""

from __future__ import annotations

import torch

WIDTH = 78


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def note(text: str) -> None:
    for line in text.strip("\n").splitlines():
        print(f"  {line}")


def receptive_fields(ks: list[int], ss: list[int]) -> list[int]:
    """逐层感受野：rf_l = rf_{l-1} + (k_l - 1) * prod(s_1..s_{l-1})"""
    rf, jump, out = 1, 1, []
    for k, s in zip(ks, ss):
        rf = rf + (k - 1) * jump
        jump *= s
        out.append(rf)
    return out


def sec_receptive() -> None:
    banner("1. 感受野：3x3 一层层堆，到底看到多大")
    depth = 6
    flat = receptive_fields([3] * depth, [1] * depth)
    down = receptive_fields([3] * depth, [2] * depth)

    note("每层都用 3x3 卷积，只改 stride：")
    print()
    print(f"    {'layer':<14}" + "".join(f"{i:>6}" for i in range(1, depth + 1)))
    print(f"    {'3x3, s=1':<14}" + "".join(f"{v:>6}" for v in flat))
    print(f"    {'3x3, s=2':<14}" + "".join(f"{v:>6}" for v in down))
    note("""
stride=1 时感受野线性长（每层才 +2），stride=2 时指数长（3, 7, 15, 31 ...）。
所以"网络更深"和"看得更广"不是同一件事：想快速扩大视野，靠的是下采样。

反过来也能算够不够：目标在输入上如果是 40 像素见方，感受野爬到 40 以上就到顶了，
再堆层对"看见它"没有增益。
""")


def sec_params() -> None:
    banner("2. 参数量：卷积 vs 全连接")
    h = w = 32
    c_in, c_out, k = 3, 64, 3
    n_in = c_in * h * w
    n_out = c_out * (h - k + 1) * (w - k + 1)
    conv_p = k * k * c_in * c_out + c_out
    fc_p = n_in * n_out + n_out

    note(f"输入 {c_in}x{h}x{w}，要得到 {c_out} 通道、{h - k + 1}x{w - k + 1} 的特征图：")
    print()
    print(f"    全连接   {n_in} x {n_out} + {n_out}")
    print(f"             = {fc_p:,} 个参数")
    print(f"    3x3 卷积 3x3x{c_in}x{c_out} + {c_out}")
    print(f"             = {conv_p:,} 个参数")
    print(f"    倍数     {fc_p / conv_p:,.0f}x")
    note("""
同一件事，卷积只要全连接参数量的约十万分之一 —— 省下来的就是参数共享。
这才是 CNN 能吃大图的原因，也是课件那句 "larger model bias for image" 的实际含义：
不是卷积更聪明，是我们先把"图里该有的规律"写进了结构。
""")


def sec_channel() -> None:
    banner("3. 通道数和感受野是两件独立的事")
    x = torch.randn(3, 8, 8)
    conv1 = torch.nn.Conv2d(3, 16, kernel_size=1)
    conv3 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

    print(f"    输入            {tuple(x.shape)}")
    print(f"    1x1 conv -> 16  {tuple(conv1(x).shape)}   参数 "
          f"{sum(p.numel() for p in conv1.parameters())}")
    print(f"    3x3 conv -> 16  {tuple(conv3(x).shape)}   参数 "
          f"{sum(p.numel() for p in conv3.parameters())}")
    note("""
两种都变成 16 通道，但 1x1 的输出只看得到 1 个像素，3x3 的看得到 3x3。

  · 通道数 = 你要几个 filter，想设多少设多少   -> 决定"看什么"
  · 感受野 = kernel 和 stride 累积出来的        -> 决定"看多大"

课件第一层常用 3x3 stride 2（或 1x1）把 RGB 的 3 通道抬到 32/64：
在参数量最便宜的地方换容量，顺手把分辨率砍半。
""")


def _rows(t: torch.Tensor) -> list[str]:
    return [" ".join(str(int(v)) for v in row) for row in t.squeeze().tolist()]


def sec_pooling() -> None:
    banner("4. pooling 的不变性是局部的，不是全局的")
    img = torch.zeros(1, 1, 8, 8)
    img[..., 2:4, 2:4] = 1.0                      # 一个 2x2 的亮块
    shifted = torch.roll(img, shifts=1, dims=-1)  # 整幅图右移 1 个像素

    pool = torch.nn.MaxPool2d(2, 2)
    gap = torch.nn.AdaptiveMaxPool2d(1)
    a, b = _rows(pool(img)), _rows(pool(shifted))

    print(f"    {'2x2 maxpool (orig)':<24}{'shift 1px'}")
    for ra, rb in zip(a, b):
        print(f"    {ra:<24}{rb}")

    d_pool = (pool(img) - pool(shifted)).abs().max().item()
    d_gap = (gap(img) - gap(shifted)).abs().max().item()
    print()
    print(f"    2x2 maxpool      最大差异 {d_pool:.1f}")
    print(f"    global max pool  最大差异 {d_gap:.1f}")
    note("""
2x2 maxpool 把"4 个格子之内的差别"压成了"1 个格子的位移"，但没有消掉差别；
global max pool 把整张图压成 1 个数，那才是真正的平移不变。

顺带两点：
  · 池化层没有参数，行为完全由超参决定（kernel / stride / max 还是 avg），确实不用学。
  · 课件最后一页说 CNN 对缩放、旋转没有不变性 —— 池化只对平移有局部容忍，
    所以这两类变换只能靠 data augmentation 塞进去。
""")


def main() -> None:
    torch.manual_seed(0)
    sec_receptive()
    sec_params()
    sec_channel()
    sec_pooling()


if __name__ == "__main__":
    main()
