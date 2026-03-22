import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SoftmaxAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        similarity = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = similarity / math.sqrt(self.d_model)
        
        attention_weight = F.softmax(scaled_scores, dim=-1)
        
        attn_output = torch.matmul(attention_weight, v)
        return self.out_proj(attn_output)

class SoftmaxTransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = SoftmaxAttention(d_model)
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
