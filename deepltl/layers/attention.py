"""Implementation of scaled dot-product attention and multi-head attention as described in 'Attention Is All You Need' (Vaswani et al., 2017) based on https://www.tensorflow.org/tutorials/text/transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(queries, keys, values, mask=None):
    """
    Args:
        queries: (..., num_queries, d_queries)
        keys: (..., num_keys, d_queries)
        values: (..., num_keys, d_values)
        mask: (..., num_queries, num_keys)
    Returns:
        attention: (..., num_queries, d_values)
        attention_weights: (..., num_queries, num_keys)
    """
    attention_logits = torch.matmul(queries, keys.transpose(-2, -1))  # (..., num_queries, num_keys)

    # scale by square root of d_queries
    d_queries = queries.size(-1)
    scaled_attention_logits = attention_logits / torch.sqrt(torch.tensor(d_queries, dtype=queries.dtype))

    # mask scaled values
    if mask is not None:
        scaled_attention_logits += (mask * torch.finfo(queries.dtype).min)

    # perform softmax over key axis and multiply resulting attention weights with values
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)  # (..., num_queries, num_keys)
    attention = torch.matmul(attention_weights, values)  # (..., num_queries, d_values)
    return attention, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_embedding, num_heads):
        if d_embedding % num_heads != 0:
            raise ValueError(f"Embedding dimension {d_embedding} must be divisible by number of heads {num_heads}.")

        super().__init__()

        self.d_embedding = d_embedding
        self.num_heads = num_heads
        self.d_heads = d_embedding // num_heads

        self.Q = nn.Linear(d_embedding, d_embedding)
        self.K = nn.Linear(d_embedding, d_embedding)
        self.V = nn.Linear(d_embedding, d_embedding)

        self.final_projection = nn.Linear(d_embedding, d_embedding)

    def _split_heads(self, input, batch_size):
        """
        Splits last dimension d_embedding into (num_heads, d_heads) and transposes result
        Args:
            input: (batch_size, num_inputs, d_embedding)
        Returns:
            (batch_size, num_heads, num_inputs, d_heads)
        """
        input = input.view(batch_size, -1, self.num_heads, self.d_heads)
        return input.transpose(1, 2)

    def forward(self, queries, keys, values, mask=None, cache=None):
        """
        Args:
            queries: (batch_size, num_queries, d_embedding)
            keys: (batch_size, num_keys, d_embedding)
            values: (batch_size, num_keys, d_embedding)
            mask: (batch_size, num_queries, num_keys)
            cache: a dictionary with attention from previous decoding steps that is used for fast decoding and has the following form:
                {'keys': [batch_size, i, num_heads, d_heads]
                 'values': [batch_size, i, num_heads, d_heads]}
                where i is the number of previous decoding steps
        Returns:
            attention: (batch_size, num_queries, d_embedding)
            attention_weights: (batch_size, num_queries, num_keys)
        """
        batch_size = queries.size(0)

        queries = self.Q(queries)
        keys = self.K(keys)
        values = self.V(values)

        queries = self._split_heads(queries, batch_size)  # (batch_size, num_heads, num_queries, d_heads)
        keys = self._split_heads(keys, batch_size)  # (batch_size, num_heads, num_keys, d_heads)
        values = self._split_heads(values, batch_size)  # (batch_size, num_heads, num_keys, d_heads)

        if cache is not None:
            # concatenate cached keys and values
            keys = torch.cat([cache['keys'].transpose(1, 2), keys], dim=2)
            values = torch.cat([cache['values'].transpose(1, 2), values], dim=2)
            # update cache
            cache['keys'] = keys.transpose(1, 2)
            cache['values'] = values.transpose(1, 2)

        scaled_attention, attention_weights = scaled_dot_product_attention(queries, keys, values, mask)  # (batch_size, num_heads, num_queries, d_heads) (batch_size, num_heads, num_queries, num_keys)
        scaled_attention = scaled_attention.transpose(1, 2)  # (batch_size, num_queries, num_heads, d_heads)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_embedding)  # (batch_size, num_queries, d_embedding)
        attention = self.final_projection(concat_attention)  # (batch_size, num_queries, d_embedding)
        return attention, attention_weights