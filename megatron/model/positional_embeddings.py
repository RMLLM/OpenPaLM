# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        cos, sin = self._get_cos_sin(self.max_position_embeddings, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if seq_len > self.max_seq_len_cached:
            cos, sin = self._get_cos_sin(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            self.cos_cached = cos
            self.sin_cached = sin
        return self.cos_cached[:seq_len, ...].to(x.dtype), self.sin_cached[:seq_len, ...].to(x.dtype)

    def _get_cos_sin(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # [sx, b, np, hn]
        return emb.cos()[:, None, None, :], emb.sin()[:, None, None, :]

# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
