import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2   ## broadcasting is involved in the following operations:
                                                                     ## self.a_2 * (x - mean): Here, self.a_2 has shape (A,) and (x - mean) has shape (B, C, A). To perform the element-wise multiplication, self.a_2 is broadcasted to shape (1, 1, A) or equivalently (B, C, A) to match the shape of (x - mean).
                                                                     ## self.b_2 is added to the result of the previous operation. Assuming self.b_2 has shape (A,), it is broadcasted to shape (B, C, A) to match the shape of the result.
