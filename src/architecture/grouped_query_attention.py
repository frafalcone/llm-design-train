import torch

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, gqa_config: dict):
        super().__init__()
        self.embedding = gqa_config.get("embedding", 0)
        self.num_heads = gqa_config.get("number_of_heads", 0)
        self.num_groups = gqa_config.get("number_of_groups", 0)
        self.max_seq_len = gqa_config.get("context_length", 0)
        self.dropout_rate = gqa_config.get("dropout_rate", 0)

        assert self.embedding % self.num_heads == 0
        assert self.num_heads % self.num_groups == 0

        self.head_dim = self.embedding // self.num_heads
        self.heads_per_group = self.num_heads // self.num_groups

        self.q_proj = torch.nn.Linear(self.embedding, self.embedding, bias=False)
        self.k_proj = torch.nn.Linear(self.embedding, self.num_groups * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.embedding, self.num_groups * self.head_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.embedding, self.embedding, bias=False)
        self.out_proj.is_residual_proj = True

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        position = torch.arange(self.max_seq_len)
        freqs = torch.outer(position, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().view(1, 1, self.max_seq_len, self.head_dim // 2))
        self.register_buffer("sin_cached", freqs.sin().view(1, 1, self.max_seq_len, self.head_dim // 2))

    def _apply_rope(self, x, seq_len):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        assert cos.shape[-1] == x.shape[-1] // 2, (
            f"RoPE dim mismatch: cos has {cos.shape[-1]} pairs, x has {x.shape[-1]} features"
        )

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

    def forward(self, x):
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.num_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_groups, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, t)
        k = self._apply_rope(k, t)

        k = k.unsqueeze(2).expand(b, self.num_groups, self.heads_per_group, t, self.head_dim).reshape(b, self.num_heads, t, self.head_dim)
        v = v.unsqueeze(2).expand(b, self.num_groups, self.heads_per_group, t, self.head_dim).reshape(b, self.num_heads, t, self.head_dim)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True,
            dropout_p=self.dropout_rate if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(b, t, self.embedding)
        return self.out_proj(out)