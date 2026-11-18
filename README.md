# Deep Learning 最小 demo 集 + 学习笔记

一套对着课件逐条验证的 PyTorch demo，配一份按技术分章的学习笔记。
每个 demo 都很小（一个文件几十行），只跑 CPU，不训练大模型 —— 目的是**把机制看清楚**，
而不是刷指标。

## 内容一览

| 主题 | demo | 关键问题 |
| --- | --- | --- |
| CNN | `cnn/` | 感受野怎么长、通道数和感受野为什么是两件事、pooling 的不变性有多局部 |
| RNN | `rnn/` | 「循环」到底指什么、单向和双向差在哪、信息衰减有多快 |
| LSTM | `lstm/` | 四个门各管什么、为什么能记长程、c 通道为什么关键 |
| Self-Attention | `attention/` | Q/K/V 怎么算、为什么它不知道顺序、位置编码怎么补 |
| Transformer | `transformer/` | FFN 干什么、BN 为什么不如 LN、pre/post-norm 差在哪 |
| Decoder / Seq2seq | `transformer/` | masked attention 为什么必须、cross-attention 的 Q 从哪来 |
| 高效 Attention | `efficient_attention/` | 五条省算力的路，哪条真能加速、哪条只是"看起来省" |
| GAN | `gan/` | 为什么需要分布、MSE 为什么会糊、两个网络怎么互相骗 |

学习笔记：[`notes/`](notes/README.md)（含 203 张课件截图和逐条讲解）

## 快速开始

```bash
python main.py               # 列出所有 demo
python main.py cnn           # 跑某一个
python main.py all           # 八个全跑
```

`python -m <主题>.demo` 也可以。环境只依赖 torch（CPU 版即可）：

```bash
python check_env.py          # 只读体检：conda / python / torch
pip install -r requirements.txt
```

## 目录结构

```
DeepLearning/
├── main.py               统一入口
├── check_env.py          环境体检（只读）
├── requirements.txt      只有 torch
│
├── cnn/                  demo.py               感受野 / 参数量 / 通道数 / pooling
├── rnn/                  cell.py, models.py, vocab.py, report.py, demo.py
├── lstm/                 cell.py, demo.py
├── attention/            model.py, demo.py
├── transformer/          norm.py, block.py, demo.py
├── efficient_attention/  model.py, demo.py
├── gan/                  model.py, divergence.py, demo.py
│
└── notes/                README.md + 01~10 章 + assets/（课件图）
```

每个包内部统一：**`model.py`（或 `cell.py`）放实现，`demo.py` 放演示**。
demo 之间不互相依赖，删掉任何一个不影响其它。

## 笔记章节

| # | 章节 | 一句话 |
| --- | --- | --- |
| 01 | [CNN](notes/01-cnn.md) | 局部连接 + 参数共享，是同一种模式在不同位置复用同一组权重 |
| 02 | [RNN](notes/02-rnn.md) | 循环 = 同一组权重沿时间反复使用，参数和序列长度无关 |
| 03 | [LSTM](notes/03-lstm.md) | c 走加法通道，所以梯度不用穿过一串 tanh 导数 |
| 04 | [Self-Attention](notes/04-self-attention.md) | 用 Q 找 K，softmax 之后对所有 V 加权求和 |
| 05 | [Transformer](notes/05-transformer.md) | attention 是通信，FFN 是计算，FFN 占参数 2/3 |
| 06 | [Decoder / Seq2seq](notes/06-decoder-seq2seq.md) | masked 遮未来、cross-attention 的 Q 来自 decoder |
| 07 | [高效 Self-Attention](notes/07-efficient-attention.md) | 五条省算力的路，以及哪条真的能加速 |
| 08 | [GAN](notes/08-gan.md) | 不定义对错，只训一个"像不像"的裁判 |
| 09 | [GAN 理论](notes/09-wgan.md) | divergence 不会算 → 用 D 去估 → JS 在重叠为 0 时失效 → WGAN |
| 10 | [序列生成 · 评估 · cGAN](notes/10-gan-seq-eval-conditional.md) | argmax 断梯度 → 用 RL；IS/FID；条件生成改的是 D |

## 几个值得记住的数字

全部可以用 `python main.py <主题>` 复现。

| 结论 | 数字 |
| --- | --- |
| 卷积 vs 全连接的参数比 | 1,792 vs 177,004,800（**98,775 倍**） |
| 3×3 堆 6 层，stride=1 的感受野 | 13（线性，`2L+1`） |
| 同样 6 层，stride=2 的感受野 | 127（指数） |
| 双向 RNN 反向通道的跨句差异 | t=2..5 **精确为 0**（还没读到第一个词） |
| 忘记门随机初始化下 8 步后的记忆残留 | **0%**（`f≈0.5`，一步砍半） |
| self-attention 打乱顺序（无 PE） | `max\|Δ\| = 0`（真的不知道顺序） |
| Transformer block 里 FFN 的参数占比 | **67%** |
| BN 的 batch 依赖程度 | LN `0` vs BN `1.8479` |
| 换乘法顺序 `(QKᵀ)V` → `Q(KᵀV)` | 相对误差 `4.4e-07`（数学等价），快 **24 倍** |
| 稀疏化：置 0 了 87% 的格子 | 耗时反而从 2.87ms 涨到 **6.33ms**（mask≠加速） |
| 随机权重下注意力矩阵保留 K=32 | 重构误差 **42%**（低秩是训练出来的，不是数学性质） |
| MSE 学双峰数据 | 输出 **0**（真实值只有 ±2，条件均值是个不存在的值） |
| GAN 两步交替后 D 的参数变化 | 训练 G 时 D 变化 **0.00e+00**（真的冻住了） |
| `max_D V(D)` vs `2·JS - 2log2` | **-0.9440 vs -0.9440**（完全相等） |
| JS 在不重叠区间内 | θ 从 10 到 6，JS 只动 **0.004**，Wasserstein 从 10 掉到 6 |
| WGAN 三种 Lipschitz 约束 | 无约束梯度 **531,864**；GP 后梯度 1.10、差值 **2.198 ≈ W1** |
| argmax 断梯度 | 第二名概率 0.316→**0.903**，D 打分变化 **0.0000000000** |
| REINFORCE 200 步 | 目标 token 概率 0.200→**0.954** |
| 模式崩塌的覆盖率 | 正常 **100%** vs 崩塌 **12%** |

## 约定

- 所有随机权重都固定 seed，数字可复现
- 不写训练循环、不用 Dataset、不加配置文件 —— 保持「一眼看完一个文件」
- 笔记里的「我本来以为…但实测…」是故意留的：结论不值钱，怎么推翻结论才值钱
