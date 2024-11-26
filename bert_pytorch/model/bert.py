import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4  ## corresponds to the third parameter of TransformerBlock.

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        ## before masking here: 'x' is of (batch_size, seq_len)
        ## after masking here:
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)    ## This line of code creates a mask for padded tokens in a sequence.
                                                                            ## Note, this is mask, not x
                                                                            ## (x > 0) creates a boolean tensor where True indicates a non-padded token (i.e., a token with an index greater than 0) and False indicates a padded token.
                                                                            ## unsqueeze(1) adds a new dimension of size 1 to the tensor, effectively making it a 3D tensor.
                                                                            ## repeat(1, x.size(1), 1) repeats the tensor along the second dimension (x.size(1)) to match the sequence length. 
                                                                            ## This creates a tensor with shape (batch_size, seq_len, seq_len).
                                                                            ## The final unsqueeze(1) adds another dimension, resulting in a tensor with shape (batch_size, 1, seq_len, seq_len).
                                                                            ## This mask is used to ignore padded tokens during self-attention computations in the Transformer model.

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
