"""Self-attention：Q/K/V、注意力矩阵、位置编码、多头。"""

from .model import SelfAttention, sinusoidal_pe

__all__ = ["SelfAttention", "sinusoidal_pe"]
