import torch
import torch.nn as nn

class QTGNResonator(nn.Module):
    def __init__(self, vocab_size, embed_dim, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projector = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, x):
        if x.numel() == 0: 
            return torch.zeros(self.projector.out_features, device=x.device)
        embedded = self.embedding(x)
        return self.projector(torch.mean(embedded, dim=0))

class QTGNDynamics(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.W_in = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_fb = nn.Linear(embed_dim, latent_dim, bias=False)
        self.gamma = nn.Parameter(torch.ones(latent_dim) * 1.0)
        
    def forward(self, t, z, context_vec, last_embed):
        decay = -self.gamma * z
        input_drive = self.W_in(context_vec)
        feedback = self.W_fb(last_embed)
        return decay + input_drive + feedback

class QTGNDecoder(nn.Module):
    def __init__(self, latent_dim, vocab_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, vocab_size)
        )
        
    def forward(self, z):
        return self.head(z)
