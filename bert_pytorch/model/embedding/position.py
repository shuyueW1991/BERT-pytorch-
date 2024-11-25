import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] 
    ## from the annotation of `# torch.ByteTensor([batch_size, 1, seq_len, seq_len)` and the following line
    ## `mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)` in class BERT in bert.py, 
    ## we come to assume that `x.size(1)` referring to seq_len. 
    ## So, back to here, the pe has its 0th dimen as 1 as batch_size (thanks to  unsqueeze(0)), 1th as seq_len, 2nd as d_model.
    