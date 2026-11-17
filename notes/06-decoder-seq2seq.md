# 06 · Decoder 与 Seq2seq

> 一句话：encoder 把源句变成一串「可查的 K/V」，decoder 用自己的 Q 一位一位地问、一位一位地生成。

## 1. 全景：谁决定输出

```
源句 → Encoder → memory（一串 K/V）
                        ↓
      BEGIN → Decoder → 機 → 器 → 學 → 習 → END
                 ↑______________________|   自回归：上一步的输出是这一步的输入
```

**要纠正一个常见说法**：「encoder 决定要输出什么样的 sequence」—— 不对。

- encoder 只负责把源句整理成 memory，它不知道要生成什么
- decoder 每一步用**自己的 Q** 去 memory 里查，然后**自己决定**下一个 token
- 决定权在 decoder，encoder 只提供「可查的键值对」

![seq2seq：encoder 产出 memory，decoder 逐位生成](assets/seq2seq/img-01.jpg)

![decoder 的输出逐位回填成下一步的输入](assets/seq2seq/img-02.jpg)

## 2. Masked Self-Attention：为什么要遮住未来

普通 self-attention 是**全连接**的：每个位置都能看到所有位置（包括后面）。

![普通 self-attention：所有位置互相可见](assets/seq2seq/img-05.jpg)

但 decoder 是**自回归**的：第 t 个 token 还没生成，怎么可能看见它？

所以要把「未来」遮掉 —— 加一个上三角的 mask，把 `j > i` 的位置填成 `-inf`，softmax 之后权重为 0：

```
. X X X
. . X X
. . . X
. . . .
```

`b¹` 只能看 `a¹`，`b³` 能看 `a¹a²a³`。这就产生了「顺序语义」：**前面的看不到后面的，后面的看得到前面的**。

![masked self-attention：只保留左下方](assets/seq2seq/img-06.jpg)

![每个位置只 attend 到自己及以前](assets/seq2seq/img-07.jpg)

> 另一种等价理解：训练时如果让第 t 位看到第 t+1 位，模型会直接抄答案，loss 掉到 0，
> 但推理时没有答案可抄 —— 训练和推理就彻底脱节了。

![decoder block：多了一个 Masked](assets/seq2seq/img-03.png)

![encoder 与 decoder 的对照](assets/seq2seq/img-04.jpg)

## 3. 什么时候停：输出长度问题

**We do not know the correct output length.** 输入的句子长度和输出长度没有固定关系。

如果什么都不做，decoder 会一直预测下去（Never stop!）。解决办法很朴素：**准备一个特殊标记 END**。

生成到 END 就停。所以词表里要多一个 token，训练数据里每条目标句都要在末尾补 END。

![END 作为一个普通 token 参与预测](assets/seq2seq/img-09.png)

## 4. AT 与 NAT

| | AT（Autoregressive） | NAT（Non-Autoregressive） |
| --- | --- | --- |
| 生成方式 | 一个一个吐 | **一次吐出整句** |
| 输入 | BEGIN + 前面已生成的 token | **全是 BEGIN**（并行） |
| 速度 | 慢（串行 L 步） | **快（1 步）** |
| 质量 | 通常更好 | 通常略差 |
| 输出长度 | 由 END 自然决定 | **要额外想办法** |

NAT 怎么定长度？两个办法：① 另训一个「长度预测器」；② 直接吐一个很长的序列，END 之后的丢掉。

![AT decoder：输入是前面已生成的 token](assets/seq2seq/img-10.png)

![NAT decoder：输入全是 BEGIN](assets/seq2seq/img-11.png)

![NAT 的输出长度怎么定](assets/seq2seq/img-12.png)

![NAT 也要有 END](assets/seq2seq/img-13.png)

## 5. Cross-Attention：Q 从哪来

这是 seq2seq 的核心，也是全篇最该记的一条：

| | Q 来自 | K、V 来自 |
| --- | --- | --- |
| decoder 的 **self**-attention | decoder 自己 | decoder 自己 |
| decoder 的 **cross**-attention | **decoder** | **encoder 的 memory** |

用形状记住它：`A` 的形状是 `(batch, heads, 目标长度, 源长度)`。
实测 `(2, 4, 3, 5)` —— 3 是目标句长度，5 是源句长度。

self-attention 那边的 `A` 是 `(2, 4, 3, 3)`，两个数字相等正是「自己查自己」的体现。

![cross attention 在 decoder 里的位置](assets/seq2seq/img-14.jpg)

![Q 来自 decoder，K/V 来自 encoder](assets/seq2seq/img-15.jpg)

![每一步的 Q 都是当前 decoder 状态](assets/seq2seq/img-16.jpg)

## 6. 训练：Teacher Forcing

训练时 decoder 的输入不是自己上一步的输出，而是**正确答案**（ground truth）—— 这叫 teacher forcing。

好处：**一次前向就能并行算完所有位置**，每个位置都算 cross entropy，加起来最小化。
坏处：模型从没见过自己的错误（见下一节）。

```
输入:  BEGIN  機   器   學   習
目标:   機    器   學   習  END
        ↑ 整体右移一位，这就叫 "Outputs (shifted right)"
```

![训练时最小化每个位置的 cross entropy](assets/seq2seq/img-23.jpg)

![teacher forcing：拿正确答案当输入](assets/seq2seq/img-24.jpg)

## 7. 训练与推理的不一致：Exposure Bias

| 阶段 | decoder 的输入 |
| --- | --- |
| 训练 | **正确答案** |
| 推理 | **自己上一步的输出** |

一旦某一步生成了错的 token，后面就全在错误的前提下继续生成 —— **一步错步步错**。
这个训练/测试分布的错配叫 **exposure bias**。

缓解办法：**Scheduled Sampling** —— 训练时按一定概率把「模型自己的预测」喂回去，
让模型提前适应「输入里有噪声」的状态。概率从 0 慢慢升到某个值，不要一上来就全用预测。

![exposure bias：训练见真值，推理见自己的输出](assets/seq2seq/img-32.jpg)

![scheduled sampling：按概率混入模型自己的预测](assets/seq2seq/img-33.jpg)

> 顺带回答你那句「训练时偶尔给一些错误的东西，反而学得更好」—— 对，这是一整类方法：
> scheduled sampling、label smoothing、dropout、输入加噪、SpecAugment。
> 共同点是**别让模型过度依赖「完美输入」**。但比例要调，加多了会掉点。

## 8. 解码策略：greedy / beam search / sampling

| 策略 | 做法 | 特点 |
| --- | --- | --- |
| Greedy | 每步取概率最大的 | 快，但局部最优 ≠ 全局最优 |
| Beam Search | 保留 top-k 条路径 | 更可能找到高分路径，但 k 大有代价 |
| Sampling | 按分布采样 | 有多样性，可能"错"但更自然 |

Beam search 是**近似搜索**：穷举所有路径是 `V^L`（爆炸），所以只保留 k 条。
而且是**训练时最大化的目标 ≠ 推理时搜索的目标**，两者不一致。

![beam search：每步保留 k 条路径](assets/seq2seq/img-27.jpg)

**关键认知：最高分的那条路不一定是最好的输出。** 具体看任务：

- 有确定答案的任务（翻译、语音识别）→ beam search 合适
- 需要创造性的任务（续写、对话、TTS）→ greedy/beam 会让输出**退化**，
  因为「最可能」往往等于「最平庸」

![beam search 的退化：反复重复同一句](assets/seq2seq/img-28.jpg)

![有些任务确实需要随机性](assets/seq2seq/img-29.png)

![接受不完美](assets/seq2seq/img-30.png)

## 9. 关于 TTS 里的 noise

你问「做 TTS 时 decoder 要加 noise，这个 noise 具体指向什么」—— 这个问题问得很好，
因为它是理解「生成任务为什么要采样」的钥匙。

**根源：训练用 L1/L2 或 mel loss，最优解是条件均值。**

给定文本，可能有无数种合理的读法（语速、停顿、重音、音色）。用回归损失训练时，
模型会学到这些可能性的**平均** —— 结果就是「平淡、没有起伏、每个字都读得一样」的声音。

noise 的作用是让模型**从分布里采样**，而不是取均值：

| noise 加在哪 | 具体形式 | 控制什么 |
| --- | --- | --- |
| 输入侧 | reference encoder / GST 的 dropout | 音色、风格 |
| 隐变量 | VAE 的 `z`（如 VITS） | 韵律、时长 |
| decoder | Tacotron 的 pre-net dropout (0.5) | 防止死盯上一帧，增加变化 |
| 采样 | temperature / top-k | 整体随机性强度 |

所以「noise 指向什么」的答案是：**它指向这个任务里「一对多」的那些自由度**。

| 任务 | noise 指向 |
| --- | --- |
| TTS | 韵律、时长、音色 |
| 文本续写 | 用词、句式、话题走向 |
| 图像生成 | 纹理、细节、构图 |
| 机器翻译 / ASR | **不该有 noise** —— 答案是唯一的，加了就是制造错误 |

判断标准只有一条：**这个任务的合理输出有几个？** 多个 → 需要采样；唯一 → 不要采样。

![优化评估指标：不知道怎么优化就上 RL](assets/seq2seq/img-31.jpg)

## 10. 评估与优化

- 训练目标：每个位置 cross entropy（teacher forcing 下）
- 评估指标：BLEU / ROUGE / WER —— 都**不可微**
- 于是有了「用 RL 直接优化指标」这条路（arXiv 1511.06732），
  以及后来的 MRT / minimum risk training

这也是 exposure bias 之外，训练目标和评估目标不一致的另一个来源。

## 11. Guided Attention

有些任务里输入和输出是**单调对齐**的（语音识别、TTS）：第 i 个输出大致对应第 i 段输入。
这时可以强行约束注意力只能沿对角线走 —— 叫 monotonic / location-aware attention。

它不能提升上限，但能**大幅降低训练难度**，避免模型学到"乱跳"的对齐。

![Guided Attention：单调对齐的任务](assets/seq2seq/img-25.png)

![注意力分数沿对角线移动](assets/seq2seq/img-26.jpg)

## 12. Q&A

**Q：decoder 每一层的输入都是 encoder 最后一层的输出吗？**
A：标准 Transformer 是。所有 decoder 层共享同一份 memory。你截的那几张
（Granularity Consistent / Parallel / Fine-Grained / Full Matching / Adaptive Matching）
都是在改这个「层间怎么连」的问题，属于改进方案。

**Q：masked 只遮 self-attention，cross-attention 要不要遮？**
A：不需要。cross-attention 的 K/V 来自完整的源句，源句本来就是已知的，
遮住反而丢信息。要遮的只有 decoder 自己的「未来」。

**Q：NAT 又快又并行，为什么没取代 AT？**
A：质量。NAT 丢掉了 token 之间的依赖，简单任务（TTS、部分翻译）能用，
复杂生成上还是打不过 AT。近年有「少步扩散 / 迭代精炼」的思路在折中这两者。

**Q：训练时 loss 很低，推理时效果差，怎么排查？**
A：先怀疑 exposure bias（训练见真值、推理见自己），再看评估指标和训练目标是否一致，
最后看 beam search 的 k 和长度惩罚是否合理。

## 13. 对应代码

```bash
python main.py transformer     # 第 4 节：encoder/decoder 形状与因果掩码
```

`transformer/block.py` 的 `DecoderBlock` 三个子层齐全（masked self-attn / cross-attn / FFN），
`MultiHeadAttention` 一个类同时支持 self 和 cross 两种用法。
