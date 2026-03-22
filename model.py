import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SigmoidAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.layerscale = nn.Parameter(1e-4 * torch.ones(d_model))
        # self.layerscale=nn.Parameter(1e-1* torch.ones(d_model))
        # self.layerscale=1
        
    def forward(self, x):
        seq_length = x.size(1)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q_norm = F.layer_norm(q, [self.d_model])
        k_norm = F.layer_norm(k, [self.d_model])
        
        similarity = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        scaled_scores = similarity / math.sqrt(self.d_model)
        
        bias = -math.log(seq_length)
        # bias=0

        biased_scores = scaled_scores + bias
        
        attention_weight = torch.sigmoid(biased_scores)
        attn_output = torch.matmul(attention_weight, v)
        
        projected_output = self.out_proj(attn_output)
        
        return projected_output * self.layerscale

class SigmoidTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.attention = SigmoidAttention(d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x
