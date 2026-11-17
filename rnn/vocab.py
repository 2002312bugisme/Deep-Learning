"""极简词表：句子 -> token -> id。没有 BPE，就是按空格切。"""

from __future__ import annotations


def tokenize(sentence: str) -> list[str]:
    """按空格切词并小写。'November 2nd' -> ['november', '2nd']"""
    return sentence.lower().split()


class Vocab:
    """按出现顺序建词表，够 demo 用。"""

    def __init__(self, sentences: list[str]) -> None:
        tokens: list[str] = []
        for s in sentences:
            for t in tokenize(s):
                if t not in tokens:
                    tokens.append(t)
        self.itos: list[str] = tokens
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, sentence: str) -> list[int]:
        return [self.stoi[t] for t in tokenize(sentence)]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]
