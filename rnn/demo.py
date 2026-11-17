"""RNN 最小 demo 入口：单向 / 双向 + 两句话的输出差距。

跑之前先看看环境：  python check_env.py
然后运行：          python run_demo.py
"""

from __future__ import annotations

import torch

from . import report as rp
from .models import BiRNN, UniRNN
from .vocab import Vocab, tokenize

SENT_A = "arrive taipei on November 2nd"
SENT_B = "leave  taipei on November 2nd"


def main() -> None:
    torch.manual_seed(42)          # 固定随机权重，保证每次跑结果一致

    vocab = Vocab([SENT_A, SENT_B])
    tokens_a, tokens_b = tokenize(SENT_A), tokenize(SENT_B)
    ids_a, ids_b = vocab.encode(SENT_A), vocab.encode(SENT_B)

    rp.banner("0. 输入与词表")
    rp.note(f"""
这里不做训练，权重是固定种子下随机初始化的一份，模型也从未见过这两句话。
所以下面的概率数字不代表语义正确性，它演示的是"信息怎么一步步流过去"。
两句话只差第一个词：arrive / leave，其余 4 个 token 完全相同。
""")
    rp.note(f"A: {tokens_a}  ->  {ids_a}")
    rp.note(f"B: {tokens_b}  ->  {ids_b}")
    rp.note(f"词表 {len(vocab)} 个 token: {vocab.itos}")

    # ---------------- 1. 单向 ----------------
    uni = UniRNN(len(vocab))
    uni_a, uni_b = uni(ids_a), uni(ids_b)

    rp.banner("1. 单向 RNN：每步存下中间结果")
    rp.note("""
同一个 cell 被反复调用，a^t 只看得见 x^1..x^t。每一步的 a^t 都被留下来，
既是下一步的输入，也是这一步 y^t 的唯一来源 —— 课件里 "store" 那条蓝箭头。
""")
    rp.show_uni(tokens_a, uni_a, vocab, "单向 · 输入 A (arrive ...)")
    uni_gap = rp.compare(tokens_a, tokens_b, uni_a, uni_b, "单向 · A vs B")

    # ---------------- 2. 双向 ----------------
    bi = BiRNN(len(vocab))
    bi_a, bi_b = bi(ids_a), bi(ids_b)

    rp.banner("2. 双向 RNN：正向和反向都喂进同一个输出")
    rp.note("""
正向通道 x^1 -> x^T，反向通道 x^T -> x^1，两套独立参数。位置 t 的输出
y^t = W_o([a_fwd^t ; a_bwd^t])，因此 t 处的 y^t 同时读到左边和右边的中间结果。
""")
    rp.show_bi(tokens_a, bi_a, vocab, "双向 · 输入 A (arrive ...)")
    bi_gap = rp.compare(tokens_a, tokens_b, bi_a, bi_b, "双向 · A vs B")

    # ---------------- 3. 怎么读这两张表 ----------------
    rp.banner("3. 两句话的差距，怎么读")
    rp.note("""
单向那张表：t=1 时 arrive vs leave 差得最狠（d = 2.96），之后每一步都还带着
差距，因为 a^t 是一路带下去的；但注意 d 一路缩到 0.11（cos 0.9955）—— 第一个
词的信息每过一步就被 tanh 压一次，这就是 RNN 的远距离信息衰减，LSTM / GRU
要解决的正是这件事。

双向那张表：a_fwd 那一列走势相同，从 t=1 起一路带着差距；a_bwd 那一列的差距
只出现在 t=1，t=2..5 精确等于 0。原因很直白：反向通道从句子末尾开始读，走到
t=2..5 时它还没碰到第一个词，两句话对它来说完全一样，状态自然一模一样；直到
最后一步读到 x^1，差距才进来。

所以双向的 y^1 同时拿到"左边（第一个词本身）"和"右边（整句后半段）"的信息，
单向的 y^1 只有第一个词。这就是右图两条蓝箭头一起汇进 y^t 的意义。

最后落到整句表示：单向取 a^T，两句话 cos 还有 0.9955，基本分不开；双向把两条
通道 mean-pool 后拼接，cos 掉到 0.7752，第一个词的差异被保住了。真去做
arrive / leave 的判断，双向这个表示明显更好用。
""")

    # ---------------- 4. 汇总 ----------------
    rp.banner("4. 整句表示差距汇总")
    rp.note("两套随机权重，同一套词表，输入只换第一个词：")
    rp.summary([("uni  (last a^T)", *uni_gap),
                ("bi   (mean-pool)", *bi_gap)])
    rp.note("""
提醒一句：这里没有训练，W_i / W_h / W_o 都是固定种子下随机初始化的。上面所有
数字说明的是"信息怎么流、差距能不能留住"，不是语义判断能力。真要做 arrive /
leave 分类，把这两句话喂进模型、接一个 loss 反传，权重才会变得有意义。
""")


if __name__ == "__main__":
    main()
