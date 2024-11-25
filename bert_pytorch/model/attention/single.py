import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn    ## torch.matmul is a batched matrix multiplication, meaning it can handle batches of matrices. It can multiply two tensors of shapes (a, b, c) and (a, c, d) to produce a tensor of shape (a, b, d), where a is the batch dimension. The @ operator is just a more concise and readable way to perform matrix multiplication, and it can handle batches just like torch.matmul.
                                                      ## the Attention class deals with one head of attention calculation. In this case, When you multiply two tensors of shapes (batch_size, heads, seq1, d_k) and (batch_size, heads, seq2, d_k) using torch.matmul, the result will have shape (batch_size, heads, seq1, seq2). The d_k dimensions are contracted, and the seq1 and seq2 dimensions are combined to form a new tensor with shape (batch_size, heads, seq1, seq2).
                                                      ## the second tensor is typically transposed to have shape (batch_size, heads, d_k, seq2) before the multiplication. This is because torch.matmul performs matrix multiplication along the last two dimensions of the input tensors.