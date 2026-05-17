import torch
import torch.nn as nn
from .architecture import QTGNResonator, QTGNDynamics, QTGNDecoder

class QTGNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, latent_dim=32):
        super().__init__()
        self.resonator = QTGNResonator(vocab_size, embed_dim, latent_dim)
        self.dynamics = QTGNDynamics(latent_dim, embed_dim)
        self.decoder = QTGNDecoder(latent_dim, vocab_size)
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

    def generate(self, prompt_tensor, device, max_time=3.0, step_size=0.2, temperature=0.4):
        self.eval()
        context_vec = self.resonator(prompt_tensor.to(device))
        
        generated_indices = []
        z_current = torch.zeros(self.latent_dim).to(device)
        last_embed = torch.zeros(self.embed_dim).to(device)
        
        num_steps = int(max_time / step_size)
        
        for i in range(num_steps):
            t_curr = i * step_size
            t_tensor = torch.tensor([t_curr], dtype=torch.float32).to(device)
            
            dzdt = self.dynamics(t_tensor, z_current, context_vec, last_embed)
            z_next = z_current + dzdt * step_size
            
            logits = self.decoder(z_next)
            probs = torch.softmax(logits / temperature, dim=-1)
            
            token_idx = torch.multinomial(probs, 1).item()
            generated_indices.append(token_idx)
            
            last_embed = self.embedding_layer(torch.tensor([token_idx]).to(device)).squeeze(0)
            z_current = z_next
            
        return torch.tensor(generated_indices)
