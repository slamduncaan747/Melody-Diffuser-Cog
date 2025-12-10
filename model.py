from torch import nn
import torch
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-8
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return (x/rms) * self.gamma

class SwiGLU(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inner_dim*2)
        self.w2 = nn.Linear(inner_dim, dim)
    def forward(self, x):
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_inner_dim, dropout=.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_inner_dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, cond=None):
        hidden = self.norm1(x)
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + self.drop1(attn_out)

        if cond is not None:
            hidden = self.norm2(x)
            cattn, _ = self.cross_attn(hidden, cond, cond, need_weights=False) 
            x = x + self.drop2(cattn)

        hidden = self.norm3(x)
        swiglu_out = self.ffn(hidden)
        x = x + self.drop3(swiglu_out)
        return x

class MelodyDiffusor(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, n_layers, n_heads, ffn_inner_dim, dropout=.25):
        super().__init__()
        self.seq_len = seq_len
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Embedding(seq_len, dim)
        self.cond_embeddings = nn.Embedding(8, dim)
        self.cond_pos_embeddings = nn.Embedding(seq_len, dim)
        
        half_dim = dim // 2
        period = 10000
        freqs = torch.exp(-math.log(period) * torch.arange(0, half_dim).float() / half_dim)
        self.register_buffer('freqs', freqs)

        self.t_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_inner_dim, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_embeddings.weight

    def time_encoder(self, t):
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, x, t, cond=None):
        B, L = x.shape
        token_emb = self.token_embeddings(x)
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embeddings(pos_ids)

        t_emb = self.time_encoder(t)
        t_emb = self.t_proj(t_emb).unsqueeze(1)
        h = token_emb + pos_emb + t_emb

        c_emb = None
        if cond is not None:
            c_emb = self.cond_embeddings(cond)
            cond_len = cond.shape[1]
            cond_pos_ids = torch.arange(cond_len, device=cond.device).unsqueeze(0).expand(B, cond_len)
            c_emb = c_emb + self.cond_pos_embeddings(cond_pos_ids)

        for block in self.blocks:
            h = block(h, c_emb)

        h = self.norm(h)
        logits = self.head(h)
        return logits
