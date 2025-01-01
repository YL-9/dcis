import math
import torch
from torch import nn

class LlamaDCISScalingRotaryEmbedding(nn.Module):
    """LlamaRotaryEmbedding extended with Divide and Conquer Incremental Search"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, each_dim_factors=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scaling_factor = scaling_factor
        self.each_dim_factors = each_dim_factors.to(device)
        self.mscale = float(0.1 * math.log(scaling_factor) + 1.0)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        dcis_freq = self.inv_freq / self.each_dim_factors
#        print(dcis_freq)

#        freqs = torch.outer(t, dcis_freq.to(t.device))
        freqs = torch.einsum("i,j->ij", t, dcis_freq.to(device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )
    '''
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )
    '''
