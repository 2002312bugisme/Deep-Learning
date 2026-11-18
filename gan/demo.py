"""GAN 最小 demo：在 2D 圆环上真训一个，看「骗」是怎么发生的。

跑法（在 D:\\Code\\DeepLearning 下）：
    D:/software/anaconda/python.exe main.py gan
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import divergence as DV
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


def sec_why() -> None:
    banner("1. 为什么需要「分布」：MSE 只会输出均值")
    torch.manual_seed(0)
    x = torch.rand(2000, 1) * 2 - 1                     # x ∈ [-1, 1]
    y = M.sample_bimodal(2000)                          # y = ±2，各一半

    net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(600):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"    训练 600 步后，同一个输入下网络的输出：")
    with torch.no_grad():
        for v in (-0.8, 0.0, 0.8):
            pred = net(torch.tensor([[v]])).item()
            print(f"      x = {v:+.1f}  ->  预测 {pred:+.3f}   "
                  f"（真实数据只有 +2 和 -2）")

    note("""
MSE 的最优解是**条件均值**：(+2 和 -2 各一半) 的均值就是 0。
所以网络学到的 0 是一个**从来没在数据里出现过的值** —— 图会糊、声音会平，都是这个原因。

这也是课件那张「同一个输入、不同输出」的意思：画画、聊天、预测视频下一帧，
正确答案本来就不是唯一的。这时候该学的是**分布**，从这个分布里采样，而不是取均值。

GAN 就是干这个的：不定义「哪个答案对」，只定义一个「像不像」的裁判。
""")


def sec_networks() -> None:
    banner("2. 两个网络，各管一件事")
    torch.manual_seed(1)
    g, d = M.Generator(), M.Discriminator()
    z = torch.randn(4, 2)

    print(f"    Generator      z {tuple(z.shape)}  ->  样本 {tuple(g(z).shape)}"
          f"   参数 {sum(p.numel() for p in g.parameters()):,}")
    print(f"    Discriminator  样本 {tuple(g(z).shape)}  ->  分数 {tuple(d(g(z)).shape)}"
          f"   参数 {sum(p.numel() for p in d.parameters()):,}")
    print()
    print(f"    未训练时：D 对真数据的平均分 {M.d_score(d, M.sample_ring(256)):.3f}，"
          f"对假数据的平均分 {M.d_score(d, g(z)):.3f}")
    note("""
两个网络的角色：

  · Generator：吃一个**低维随机向量** z（从简单分布采），吐一个**高维样本** y。
    它自己不知道什么是「真」，只负责把 z 铺到数据空间里。
  · Discriminator：吃一个样本，吐一个标量分数。它也不知道怎么生成，只负责挑刺。

注意 D 的输出是 logit（可正可负），代码里用 sigmoid 换算成概率才打印出来。
未训练时两者都是乱猜，分数接近 0.5 —— 这很正常。
""")


def sec_two_steps() -> None:
    banner("3. 「骗」到底是怎么操作的：两步交替")
    torch.manual_seed(2)
    g, d = M.Generator(), M.Discriminator()
    bce = nn.BCEWithLogitsLoss()
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-3, betas=(0.5, 0.999))
    real = M.sample_ring(256)

    # ---- 第一步：冻结 G，只更新 D ----
    print("    第一步  Fix G, update D")
    g_snap = M.params_snapshot(g)
    for _ in range(300):
        z = torch.randn(128, 2)
        fake = g(z).detach()                      # 关键：detach，梯度不回流到 G
        loss = bce(d(real), torch.ones(256, 1)) + bce(d(fake), torch.zeros(128, 1))
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
    print(f"      G 参数变化 {M.params_changed(g, g_snap):.2e}   ← 应该是 0，G 没动")
    print(f"      D 对真数据的分数 {M.d_score(d, real):.3f}")
    print(f"      D 对假数据的分数 {M.d_score(d, g(torch.randn(256, 2))):.3f}")

    # ---- 第二步：冻结 D，只更新 G ----
    print()
    print("    第二步  Fix D, update G")
    d_snap = M.params_snapshot(d)
    z_fixed = torch.randn(256, 2)
    before = M.d_score(d, g(z_fixed))
    for _ in range(300):
        z = torch.randn(128, 2)
        # 把 G 和 D 接成一个网络：z -> G -> D -> 分数
        # 目标是让分数变大，所以 loss 用「骗 D 说是真的」
        loss = bce(d(g(z)), torch.ones(128, 1))
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
    after = M.d_score(d, g(z_fixed))
    print(f"      D 参数变化 {M.params_changed(d, d_snap):.2e}   ← 应该是 0，D 没动")
    print(f"      G 输出的平均分数：{before:.3f}  ->  {after:.3f}   ← 骗术在进步")

    note("""
你说的那个「接成 large network、只调 G」完全正确。代码里就是这一行：

    loss = bce(d(g(z)), ones)        # 把 G 的输出喂给 D，标签谎报成「真」

梯度会一路传回 G，但因为 **只把 G 的参数交给优化器**（`opt_g` 只含 g.parameters()），
D 一个数都不会变 —— 上面打印的「D 参数变化 0.00e+00」就是这件事的验证。

几个容易漏的实现细节：

  1. 第一步里 `fake = g(z).detach()` —— 不 detach 的话，D 的梯度会顺手改掉 G，
     那就不是「Fix G」了。
  2. G 的 loss 用的是 `bce(d(g(z)), 1)`，这叫**非饱和损失**。
     原始 GAN 论文写的是 `min log(1 - D(G(z)))`，那个形式在 D 很强时梯度会消失，
     工程上几乎都用这个等价改写。
  3. 每步 D 和 G 各更新一次，是「一步对一步」；也有 5:1 之类的比例，属于调参。
""")


def sec_result() -> None:
    banner("4. 交替训练：圆环能不能画出来")
    torch.manual_seed(3)
    g, d = M.Generator(), M.Discriminator()
    bce = nn.BCEWithLogitsLoss()
    opt_d = torch.optim.Adam(d.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(g.parameters(), lr=1e-3, betas=(0.5, 0.999))

    print("    训练前：")
    print()
    print(M.ascii_plot(M.sample_ring(160), g(torch.randn(160, 2)), w=52, h=18))
    print()

    for step in range(1, 2001):
        z = torch.randn(128, 2)
        real = M.sample_ring(128)
        fake = g(z).detach()
        loss_d = bce(d(real), torch.ones(128, 1)) + bce(d(fake), torch.zeros(128, 1))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        z = torch.randn(128, 2)
        loss_g = bce(d(g(z)), torch.ones(128, 1))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if step in (500, 1000, 2000):
            print(f"    step {step:<5} D(真) {M.d_score(d, real):.3f}"
                  f"   D(假) {M.d_score(d, g(torch.randn(256, 2))):.3f}")

    print()
    print("    训练后（o = 真数据，x = 生成，@ = 重合）：")
    print()
    print(M.ascii_plot(M.sample_ring(160), g(torch.randn(160, 2)), w=52, h=18))

    note("""
两个分数看的是「平衡」：
  · D(真) 接近 1、D(假) 接近 0  → D 完全压制 G，G 学不到东西
  · 两个都在 0.5 附近           → 势均力敌，这才是训练健康的状态

这就是 "adversarial" 这个词的来源：不是谁赢，是要一直僵持。
""")


def sec_unstable() -> None:
    banner("11. 诚实说：GAN 不好训")
    note("""
上面那个圆环是挑过的结果。GAN 的常见故障：

  · **模式崩塌（mode collapse）**：G 只学会画圆环上的一个点，反正 D 也分不出来。
    检测方法：看生成样本的多样性（覆盖了几个角度扇区）。
  · **D 太强**：G 的梯度消失，loss 看着在降但样本没变好。
  · **D 太弱**：G 随便画都能骗过 D，学不到真实分布。
  · **对超参敏感**：lr、betas、D/G 更新比例，换一组数据就得重调。

所以后来才有了 WGAN（换距离度量）、StyleGAN（换生成器结构）、
以及 diffusion 这条路线 —— 不是因为 GAN 不行，是因为它**太依赖对抗平衡**。
""")


def sec_objective() -> None:
    banner("5. 到底 minimize 什么：G* = arg min Div，而 D 的 max 值就是 JS")
    edges = torch.linspace(-6, 6, 241)
    p = DV.normal_pmf(0.0, 1.0, edges)                # P_data
    q = DV.normal_pmf(1.5, 1.0, edges)                # P_G

    d_star = DV.optimal_d(p, q)
    v_max = DV.v_value(p, q, d_star)
    v_min = DV.v_value(p, q, DV.worst_d(p, q))
    js = DV.js(p, q)

    print("    P_data ~ N(0,1)，P_G ~ N(1.5,1)，在同一个 bin 网格上离散化：")
    print()
    print(f"      JS(P_G || P_data)          = {js:.4f}")
    print(f"      2·JS - 2log2               = {2 * js - 2 * DV.LOG2:.4f}")
    print(f"      max_D V(D)  （最优 D 处）    = {v_max:.4f}   ← 和上面一致")
    print(f"      min_D V(D)  （最差 D 处）    = {v_min:.4f}")

    note("""
普通任务：`w*, b* = arg min L`，L 是人定的（MSE、交叉熵）。
生成任务：`G* = arg min Div(P_G, P_data)` —— 要 minimize 的是**两个分布的差异**。

但 Div 不知道怎么写。GAN 的办法是把它**换成一个可算的东西**：

    Div(P_G, P_data) ≈ max_D V(D, G)
    V(D, G) = E_{x~P_data}[log D(x)] + E_{x~P_G}[log(1 - D(x))]

所以总目标是：

    G* = arg min_G  max_D  V(D, G)

这就是那个 min-max。注意上面两行数字：一个具体的 D（最优那个）代进 V，
算出来的值精确等于 `2·JS - 2log2` —— 课件说「V 的最大值和 JS 有关」，这里是可验证的。

**为什么 D 要 max？** 因为 V 是一个「下界」：给定任意 D，V 都 ≤ 真实差异。
把 D 训到最好，这个下界才贴紧真值。D 没训好，量出来的差异就是错的。

一个小细节：理想的最优 D 是**逐点的** `D*(x) = p_data(x) / (p_data(x) + p_G(x))`，
不是 0 或 1。所以「D 输出 1.0 表示绝对真」只在有限容量下成立，理论最优是个概率。
""")


def sec_js_plateau() -> None:
    banner("6. JS 什么时候失效：不重叠时它是个常数")
    edges = torch.linspace(-20, 20, 2001)
    dx = (edges[1] - edges[0]).item()
    p = DV.normal_pmf(0.0, 1.0, edges)

    print(f"    {'θ':>5}{'JS':>10}{'Wasserstein':>14}{'dJS/dθ':>10}  说明")
    print("    " + "-" * 62)
    prev_js = None
    for theta in (10.0, 6.0, 4.0, 2.0, 1.0, 0.0):
        q = DV.normal_pmf(theta, 1.0, edges)
        js, w = DV.js(p, q), DV.w1(p, q, dx)
        slope = "—" if prev_js is None else f"{(js - prev_js[0]) / (theta - prev_js[1]):+.4f}"
        tag = "完全不重叠" if theta >= 4 else ("快重叠了" if theta <= 1 else "部分重叠")
        print(f"    {theta:>5.1f}{js:>10.4f}{w:>14.4f}{slope:>10}  {tag}")
        prev_js = (js, theta)

    note(f"""
这张表是整节课的转折点。

两个高斯中心离得越远，`Wasserstein` 就**线性变大**（θ=10 时是 10），但 `JS` 一路
贴着 `log2 = {DV.LOG2:.4f}` 不动 —— 斜率几乎是 0。

为什么？因为两个分布**不重叠**时，最优 D 能把它们完美分开（real 全判 1、fake 全判 0），
此时 `V = -2log2`，反推出的 `JS = log2`，**和它们离多远完全无关**。
θ=10 和 θ=6 的 G，在 JS 眼里「一样差」，梯度也是 0 —— G 收不到任何改进信号。

课件那句「JS divergence is always log2 if two distributions do not overlap」就是这个意思。

而实际训练里「不重叠」是常态，有两个原因：
  1. 数据本身是**高维空间里的低维流形**，两条流形几乎不可能相交
  2. 采样有限，就算理论上有重叠，采到的点也碰不上

所以 WGAN 换掉了 JS —— 换成 Wasserstein，它不重叠也照样给梯度。
""")


def _train_d(kind: str, real: torch.Tensor, fake: torch.Tensor, steps: int = 600):
    """训一个 1D 判别器，三种约束方式：无 / 权重裁剪 / 梯度惩罚。"""
    torch.manual_seed(0)
    d = nn.Sequential(nn.Linear(1, 64), nn.LeakyReLU(0.2),
                      nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
    opt = torch.optim.Adam(d.parameters(), lr=5e-3)

    for _ in range(steps):
        loss = -(d(real).mean() - d(fake).mean())      # WGAN 的 D 目标：拉大差值
        if kind == "gp":
            idx = torch.randint(0, real.shape[0], (64,))
            alpha = torch.rand(64, 1)
            x_hat = (alpha * real[idx] + (1 - alpha) * fake[idx]).requires_grad_(True)
            grad = torch.autograd.grad(d(x_hat).sum(), x_hat, create_graph=True)[0]
            loss = loss + 10.0 * ((grad.norm(dim=1) - 1) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if kind == "clip":
            with torch.no_grad():
                for prm in d.parameters():
                    prm.clamp_(-0.05, 0.05)
    return d


def sec_wgan() -> None:
    banner("7. WGAN：为什么要 Lipschitz 约束")
    real = torch.randn(256, 1) * 0.02                  # 真数据挤在 0
    fake = 2.0 + torch.randn(256, 1) * 0.02            # 生成的全在 2，完全不重叠
    print("    真实数据集中在 0，生成的全在 2 —— 两个分布完全不重叠")
    print(f"    真实的 1-Wasserstein 应该等于 2.00")
    print()
    print(f"    {'约束':<16}{'max|dD/dx|':>14}{'D(真)-D(假)':>16}   说明")
    print("    " + "-" * 66)
    for kind, label in (("none", "无约束"), ("clip", "权重裁剪 ±0.05"),
                        ("gp", "梯度惩罚 (GP)")):
        d = _train_d(kind, real, fake)
        x = torch.linspace(-0.5, 2.5, 200).unsqueeze(1).requires_grad_(True)
        g = torch.autograd.grad(d(x).sum(), x)[0]
        slope = g.abs().max().item()
        gap = (d(real).mean() - d(fake).mean()).item()
        memo = {"none": "D 变成陡峭的阶跃",
                "clip": "有界，但不等 W1",
                "gp": "梯度≈1，差值≈W1"}[kind]
        print(f"    {label:<14}{slope:>14.2f}{gap:>16.3f}   {memo}")

    note("""
WGAN 的目标换成了：

    max_{D ∈ 1-Lipschitz}  E_pdata[D(x)] - E_pG[D(x)]

关键在 `D ∈ 1-Lipschitz` 这个约束 —— 没有它，D 只要把真数据拉到 +∞、
假数据拉到 -∞，差值就无限大，训练根本不会收敛（上面第一行的 max|dD/dx| 就是证据）。

三种实现约束的办法：

  · **权重裁剪**（原版 WGAN）：每步把参数夹到 [-c, c]。简单，但得到的 Lipschitz 常数
    是个说不清的数 —— 所以上面那一行的差值不等于真实的 W1=2，是个被压小的值。
  · **梯度惩罚**（WGAN-GP）：直接罚 `(||∇D|| - 1)²`，让梯度贴着 1。
    这正是上面第三行能给出 ≈2.2 的原因 —— 约束对了，`E[D(真)]-E[D(假)]` 才真的接近 W1=2。
  · **谱归一化**（SN-GAN）：把每层权重矩阵的最大奇异值归一化，让整体 Lipschitz ≤ 1。

所以「WGAN 能算 Wasserstein 距离」这句话，前提是约束**真的**把 Lipschitz 常数压到 1。
用权重裁剪只能算个大概，这就是后来 GP 成为主流的直接原因。
""")



def sec_sequence() -> None:
    banner("8. 序列生成：argmax 把梯度掐断了，所以要用 RL")
    V, d = 5, 8
    torch.manual_seed(0)
    emb = torch.randn(V, d)                       # 固定词向量表
    dnet = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
    for p in dnet.parameters():
        p.requires_grad_(False)                   # D 冻结，只看 G 这边

    theta = torch.randn(V, requires_grad=True)    # decoder 输出的 logits
    probs = torch.softmax(theta, dim=0)
    order = probs.argsort(descending=True)
    first, second = int(order[0]), int(order[1])
    hard = torch.zeros(V)
    hard[first] = 1.0

    score0 = dnet(hard @ emb).squeeze()
    print(f"    当前最可能的 token = {first}，概率 {probs[first]:.3f}")
    print(f"    D 对它打分：{score0.item():+.4f}")

    # 把第二名的概率抬高一大截，但仍然是第二名
    delta = torch.zeros(V)
    delta[second] = 3.0
    probs2 = torch.softmax(theta.detach() + delta, dim=0)
    hard2 = torch.zeros(V)
    hard2[first] = 1.0
    score2 = dnet(hard2 @ emb).squeeze()
    print()
    print(f"    把第二名的 logit +3：概率 {probs[second]:.3f} -> {probs2[second]:.3f}"
          f"（第一名没变）")
    print(f"    但 argmax 还是 {first}，D 的打分变化 = {abs(score2.item() - score0.item()):.10f}")

    # hard 路径的梯度
    try:
        score0.backward()
        g_hard = theta.grad.abs().max().item()
        print(f"\n    hard 路径（argmax → one-hot）：梯度 {g_hard:.2e}")
    except RuntimeError:
        print("\n    hard 路径（argmax → one-hot）：计算图直接断掉，梯度不存在")

    # soft 路径：用概率加权（期望 embedding）
    theta2 = theta.detach().clone().requires_grad_(True)
    p2 = torch.softmax(theta2, dim=0)
    s_soft = dnet(p2 @ emb).squeeze()
    s_soft.backward()
    print(f"    soft 路径（概率加权）：梯度 {theta2.grad.abs().max():.2e}   ← 有梯度了")

    note("""
这一段就是你说的那件事，现在有数字了。

**问题**：decoder 输出 logits → softmax → **argmax** → 一个离散 token → D 打分。
argmax 是一个**分段常数**函数：只要第一名没换人，输入怎么变，输出都一模一样。
所以概率从 0.3 挪到 0.9、score 一点不动，`∂score/∂θ = 0` —— **没有任何可学的信号**。

而且这个"断"不是实现问题，是数学问题：argmax 的雅可比处处为 0（除了测度零的边界点）。

**为什么 soft 路径不能直接用**：用概率加权得到的是「期望 embedding」，
它不是任何一个真实 token —— 拿它去骗 D 是可以传梯度，但生成出来的东西不对。
（straight-through / Gumbel-softmax 是常用的折中：前向用 hard、反向用 soft 的梯度。）

**RL 的解法**：绕开 argmax，直接用 **log 概率** 的梯度。

    loss = - reward × log p(token)
    reward = D(生成的序列)

采样代替 argmax（采到哪个就用哪个 token），`log p` 对 θ 可导，reward 只当常数。
这就是 REINFORCE / policy gradient。代价是**方差大**（一次只采一个样本），
所以要 baseline、要多次采样 —— 这也解释了为什么「RL 难训 + GAN 难训」叠在一起会炸。
""")

    # REINFORCE：真跑几步，看概率往高分 token 集中
    theta3 = torch.randn(V, requires_grad=True)
    opt = torch.optim.SGD([theta3], lr=0.5)
    target = int(dnet(emb).squeeze().argmax())     # D 最喜欢哪个 token
    for _ in range(200):
        p3 = torch.softmax(theta3, dim=0)
        idx = torch.multinomial(p3, 1)
        reward = dnet(emb[idx]).squeeze().detach()
        loss = -(reward * torch.log_softmax(theta3, dim=0)[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    p_final = torch.softmax(theta3.detach(), dim=0)
    print(f"    REINFORCE 200 步后：D 最喜欢的 token 是 {target}，"
          f"它的概率从 {1.0 / V:.3f} 涨到 {p_final[target]:.3f}")
    print(f"    分数：初始 {dnet(emb[torch.randint(0, V, (1,))]).mean().item():+.3f}"
          f"  ->  现在 {dnet(emb[target]).item():+.3f}")


def sec_eval() -> None:
    banner("9. 评估：质量看单张、多样性看覆盖")
    torch.manual_seed(0)
    real = M.sample_ring(400)

    good = M.sample_ring(400)                                     # 覆盖整个环
    collapsed = M.sample_ring(400) * 0 + torch.tensor([1.0, 0.0])  # 全挤在一个点
    collapsed = collapsed + torch.randn(400, 2) * 0.02

    def coverage(x, nbins=16):
        ang = torch.atan2(x[:, 1], x[:, 0])
        idx = ((ang + 3.14159) / (2 * 3.14159) * nbins).long().clamp(0, nbins - 1)
        hit = torch.bincount(idx, minlength=nbins) > 0
        return hit.float().mean().item()

    print(f"    {'生成结果':<14}{'覆盖扇区':>10}{'平均半径':>10}{'半径标准差':>12}")
    print("    " + "-" * 50)
    for name, x in (("正常", good), ("模式崩塌", collapsed)):
        r = x.norm(dim=1)
        print(f"    {name:<14}{coverage(x):>9.0%}{r.mean():>11.3f}{r.std():>12.3f}")

    note("""
两个指标测的是**不同的东西**：

  · **质量（单张像不像）**：拿一个现成的分类器（Inception/VGG）看它的输出分布，
    集中 = 清晰、像某类 → IS 里的那部分
  · **多样性（覆盖广不广）**：把所有生成样本的预测分布**平均**，越均匀越多样
    → 这里用「覆盖了几个角度扇区」代替

坑在于：**模式崩塌时，质量指标反而可能更高**。上面第二行，所有点挤在同一个位置，
每一张都"很标准"，分类器非常确信 —— 但它只会画这一种。

所以：
  · **IS 只看单张的置信度**，不看覆盖 → 崩塌了也高（这是 IS 最大的缺陷）
  · **FID 比较两组样本的整体分布**，能捕捉崩塌 → 更常用
  · 但 FID 也有坑：它假设特征服从高斯；样本量少时估计有偏；而且**记忆 GAN**
    （直接记住训练集）能刷出极低的 FID —— 指标低不等于生成得好。
""")


def _train_cycle(lam: float, steps: int = 1500):
    """两个 1D 线段之间的无配对映射。lam=0 表示只用对抗损失。"""
    torch.manual_seed(1)
    n = 256
    t = torch.rand(n, 1) * 2 - 1
    x_dom = torch.cat([t, torch.zeros_like(t)], dim=1)        # X：x 轴上的线段
    y_dom = torch.cat([torch.zeros_like(t), t], dim=1)        # Y：y 轴上的线段

    def net():
        return nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                             nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def disc():
        return nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2),
                             nn.Linear(64, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1))

    g_xy, g_yx, d_x, d_y = net(), net(), disc(), disc()
    bce = nn.BCEWithLogitsLoss()
    o_g = torch.optim.Adam(list(g_xy.parameters()) + list(g_yx.parameters()),
                           lr=2e-3, betas=(0.5, 0.999))
    o_d = torch.optim.Adam(list(d_x.parameters()) + list(d_y.parameters()),
                           lr=2e-3, betas=(0.5, 0.999))

    for _ in range(steps):
        fx, fy = g_yx(y_dom).detach(), g_xy(x_dom).detach()
        loss_d = (bce(d_y(y_dom), torch.ones(n, 1)) + bce(d_y(fy), torch.zeros(n, 1))
                  + bce(d_x(x_dom), torch.ones(n, 1)) + bce(d_x(fx), torch.zeros(n, 1)))
        o_d.zero_grad()
        loss_d.backward()
        o_d.step()

        fake_y, fake_x = g_xy(x_dom), g_yx(y_dom)
        adv = bce(d_y(fake_y), torch.ones(n, 1)) + bce(d_x(fake_x), torch.ones(n, 1))
        cyc = ((g_yx(fake_y) - x_dom) ** 2).mean() + ((g_xy(fake_x) - y_dom) ** 2).mean()
        loss_g = adv + lam * cyc
        o_g.zero_grad()
        loss_g.backward()
        o_g.step()

    with torch.no_grad():
        out = g_xy(x_dom)
        back = g_yx(out)
        cyc_err = (back - x_dom).norm(dim=1).mean().item()
        corr = torch.corrcoef(torch.stack([t.squeeze(1), out[:, 1]]))[0, 1].item()
        spread = out[:, 1].std().item()
    return cyc_err, corr, spread


def sec_cycle() -> None:
    banner("10. Cycle GAN：没有配对数据怎么学映射")
    note("""
任务：把 X 域（x 轴上一段线）映射到 Y 域（y 轴上一段线），**没有任何配对样本**。
两个域的点各自独立采样，只知道"这群属于 X、那群属于 Y"。

只有对抗损失时会发生什么？G 只要把输出丢到 Y 域上就能骗过 D，
**输入是什么完全无所谓** —— 这正是课件里那个红色叉号「ignore input」。
""")

    print(f"    {'训练方式':<22}{'往返误差':>12}{'输入-输出相关性':>18}{'输出分散度':>12}")
    print("    " + "-" * 66)
    for lam, label in ((0.0, "只有对抗损失"), (10.0, "对抗 + cycle (λ=10)")):
        err, corr, spread = _train_cycle(lam)
        print(f"    {label:<20}{err:>12.3f}{corr:>16.3f}{spread:>12.3f}")

    note("""
三个指标，读法：

  · **往返误差** `‖G_YX(G_XY(x)) - x‖`：转过去再转回来，还能不能回到原位
  · **输入-输出相关性**：输入 x 的位置和输出 y 的位置是否还对应
  · **输出分散度**：输出在 Y 域上铺得够不够开（均匀分布的理论值是 0.577）

**先说实话：结果和我预期的不一样。** 只有对抗损失时，相关性已经到 -0.997、
分散度 0.567 —— G 自己就学到了近似单调的映射，**没有**出现课件里那种
"ignore input" 的崩塌。加 cycle 只是把往返误差从 0.057 压到 0.033。

为什么？因为 X 和 Y 都是一维线段，"均匀铺开 + 保持顺序"恰好是**最省力**的解。
梯度下降的归纳偏置在这个干净空间里自然选到了对的。

那真实任务为什么必须加 cycle？因为真实数据有**内容歧义**：
一匹马可以对应无数张斑马，而 D 只检查"输出的边际分布像不像 Y"，
管不了"这匹马变成了哪匹斑马"。这时对抗损失对 `P(y|x)` 完全没有约束，
G 完全可以选一个语义乱掉、但边际分布正确的映射。

这个 2D toy 恰好没有这种歧义，所以它**验证不了**课件里的惨状，
只能验证 cycle 项确实在压可逆性。这是一个诚实的负面结果：

    **玩具实验证不了的东西，别假装它证了。**

Cycle GAN 的 loss：

    L = L_GAN(G, D) + λ · ‖G_YX(G_XY(x)) - x‖

λ 一般取 10（原论文）。太大就只顾重建、不管像不像；太小就退化成无 cycle。

还有一点必须记住：**cycle 只保证可逆，不保证映射"对"**。
多个可逆映射都能通过这个损失，它不会告诉你哪一个是语义正确的那一个。
""")


def main() -> None:
    torch.manual_seed(0)
    sec_why()
    sec_networks()
    sec_two_steps()
    sec_result()
    sec_objective()
    sec_js_plateau()
    sec_wgan()
    sec_sequence()
    sec_eval()
    sec_cycle()
    sec_unstable()


if __name__ == "__main__":
    main()
