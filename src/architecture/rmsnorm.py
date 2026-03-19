import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, rmsn_configuration: dict):
        super().__init__()
        self.embedding = rmsn_configuration.get("embedding", 0)
        self.epsilon = rmsn_configuration.get("epsilon", 0)
        self.weight = torch.nn.Parameter(torch.ones(self.embedding))
        
    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        
        ms = x_float.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(ms + self.epsilon)
        
        x_normed = (x_float * inv_rms).to(input_dtype)
        return x_normed * self.weight