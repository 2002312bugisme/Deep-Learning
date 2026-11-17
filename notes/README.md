# 笔记索引

按技术分章，每章包含：课件要点 → 关键公式 → 我们实测的数字 → 常见误解 → 对应代码。

| 章节 | 内容 | 代码 |
| --- | --- | --- |
| [01 · CNN](01-cnn.md) | 局部连接 + 参数共享、感受野、通道数、pooling、深度判据 | `cnn/demo.py` |
| [02 · RNN](02-rnn.md) | 循环的本质、单向/双向、三根轴、信息衰减 | `rnn/demo.py` |
| [03 · LSTM](03-lstm.md) | 四个门 + c 通道、三套记法、忘记门实测 | `lstm/demo.py` |
| [04 · Self-Attention](04-self-attention.md) | Q/K/V、注意力矩阵、位置编码、多头、速度 | `attention/demo.py` |
| [05 · Transformer](05-transformer.md) | FFN、BN vs LN、pre/post-norm、encoder-decoder 层连接 | `transformer/demo.py` |
| [06 · Decoder 与 Seq2seq](06-decoder-seq2seq.md) | masked attention、AT/NAT、cross-attention、teacher forcing、beam search | `transformer/block.py` |

## 每章都有一节「Q&A」

那是把学习过程中真实卡住的问题记下来的地方，包括当时理解错、后来被数据纠正的部分。
比结论更有价值的是**怎么发现结论不成立**，所以有些地方保留了「我本来以为…但实测…」的过程。

## 图片

`assets/` 下按章节存放课件截图（共 89 张），文件名按上传顺序编号，正文里按讲解顺序引用。

```
assets/
├── cnn/          14 张
├── rnn/           3 张
├── lstm/          5 张
├── attention/    25 张
├── transformer/   8 张
└── seq2seq/      34 张
```

## 阅读顺序建议

如果是第一次过这套内容，建议按 01 → 02 → 03 → 04 → 05 → 06 走。
如果只想补某一块，04 和 06 是信息密度最高的两章（attention 的机制 + decoder 的生成策略）。

## 关于数字

笔记里所有实测数字都能用 `python main.py <主题>` 复现：

- 随机权重的地方都固定了 seed
- 只有 `attention` 第 5 节的耗时会有波动（CPU 计时），趋势稳定、绝对值 ±20%
