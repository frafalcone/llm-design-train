import torch
import torch.utils.checkpoint

from architecture.grouped_query_attention import GroupedQueryAttention
from architecture.feedforward import FeedForward
from architecture.rmsnorm import RMSNorm

class TransformerDecoder(torch.nn.Module):
    def __init__(self, trf_configuration: dict):
        super().__init__()

        gqa_config   = trf_configuration.get("gqa_configuration", {})
        ffn_config   = trf_configuration.get("ffn_configuration", {})
        rmsn_config  = trf_configuration.get("rmsn_configuration", {})

        dropout_p = gqa_config.get("dropout_rate", 0)

        self.attention    = GroupedQueryAttention(gqa_config)
        self.feedforward  = FeedForward(ffn_config)

        self.norm_1 = RMSNorm(rmsn_config)
        self.norm_2 = RMSNorm(rmsn_config)

        self.resid_dropout_1 = torch.nn.Dropout(dropout_p)
        self.resid_dropout_2 = torch.nn.Dropout(dropout_p)

        self.use_checkpoint = False

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout_1(self.attention(self.norm_1(x)))
        x = x + self.resid_dropout_2(self.feedforward(self.norm_2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)