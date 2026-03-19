import torch
 
class FeedForward(torch.nn.Module):
    def __init__(self, ffn_configuration: dict):
        super().__init__()
 
        self.embedding = ffn_configuration.get("embedding", 0)
        self.dropout_rate = ffn_configuration.get("dropout_rate", 0)
        self.bias = ffn_configuration.get("bias", False)
 
        expansion_rate = ffn_configuration.get("embedding_expansion_rate", 0)
 
        hidden_dim = int(self.embedding * expansion_rate * 2 / 3)
        hidden_dim = 8 * ((hidden_dim + 8 - 1) // 8)
 
        self.gate_proj = torch.nn.Linear(self.embedding, hidden_dim, bias=self.bias)
        self.up_proj   = torch.nn.Linear(self.embedding, hidden_dim, bias=self.bias)
        self.down_proj = torch.nn.Linear(hidden_dim, self.embedding, bias=self.bias)
        self.down_proj.is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)