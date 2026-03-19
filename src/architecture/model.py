import torch

from architecture.transformer_decoder import TransformerDecoder
from architecture.rmsnorm import RMSNorm

class Model(torch.nn.Module):
    def __init__(self, model_configuration: dict):
        super().__init__()

        self.embedding = model_configuration.get("embedding", 0)
        self.vocabulary = model_configuration.get("vocabulary", 0)
        self.dropout_rate = model_configuration.get("dropout_rate", 0)
        self.layer = model_configuration.get("layer", 0)
        self.bias = model_configuration.get("bias", False)
        self.trf_configuration = model_configuration.get("trf_configuration", {})
        self.rmsn_configuration = model_configuration.get("rmsn_configuration", {})

        self.tok_emb = torch.nn.Embedding(self.vocabulary, self.embedding)
        self.out_head = torch.nn.Linear(self.embedding, self.vocabulary, self.bias)

        self.trf_blocks = torch.nn.Sequential(*[TransformerDecoder(self.trf_configuration) for _ in range(self.layer)])
        self.final_norm = RMSNorm(self.rmsn_configuration)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.out_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self):
        for block in self.trf_blocks:
            block.use_checkpoint = True

    def gradient_checkpointing_disable(self):
        for block in self.trf_blocks:
            block.use_checkpoint = False

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            std = 0.02
            if hasattr(module, 'is_residual_proj'):
                std *= (2 * self.layer) ** -0.5
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(self, x):
        x = self.tok_emb(x)
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits