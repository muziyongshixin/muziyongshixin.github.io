import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, dim: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim

        self.query_proj = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.key_proj = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.value_proj = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        '''
        x.shape = b s d
        mask: bool tensor, True = masked (ignore).
              shape broadcasts to [b, 1, s, s] or [b, h, s, s]
        '''
        b, s, d = x.shape
        query = self.query_proj(x)  # b s (h*d)
        key = self.key_proj(x)
        value = self.value_proj(x)

        m_query = query.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # b h s d
        m_key = key.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # b h s d
        m_value = value.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # b h s d

        attention = torch.matmul(
            m_query, m_key.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            attention = attention.masked_fill(mask, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, m_value)
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out
