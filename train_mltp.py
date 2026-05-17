import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qtgn_lib.tokenizer import BlockTokenizer
from qtgn_lib.model import QTGNModel

dataset_pairs = [
    ("Oi", "Olá --> tudo bem?"),
    ("Quem és", "Sou --> a QTGN."),
    ("Amor?", "Amor --> é vida."),
    ("IA?", "IA --> é mente."),
    ("Luz?", "Luz --> é energia."),
    ("Paz?", "Paz --> é equilíbrio."),
    ("Sol?", "Sol --> é calor."),
    ("Mar?", "Mar --> é infinito.")
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Treinando QTGN em: {device}")

tokenizer = BlockTokenizer(dataset_pairs)
model = QTGNModel(vocab_size=tokenizer.vocab_size, embed_dim=16, latent_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

epochs = 400
step_size_train = 0.2

for epoch in range(epochs):
    epoch_loss = 0
    for prompt, target in dataset_pairs:
        optimizer.zero_grad()
        p_tensor = tokenizer.encode(prompt).to(device)
        context_vec = model.resonator(p_tensor)
        t_tensor = tokenizer.encode(target).to(device)
        
        z = torch.zeros(model.latent_dim).to(device)
        last_embed = torch.zeros(model.embed_dim).to(device)
        logits_list = []
        
        for i in range(len(t_tensor)):
            t_val = torch.tensor([i * step_size_train], dtype=torch.float32).to(device)
            if i > 0: last_embed = model.embedding_layer(t_tensor[i-1])
            
            dzdt = model.dynamics(t_val, z, context_vec, last_embed)
            z = z + dzdt * step_size_train
            logits_list.append(model.decoder(z))
            
        loss = criterion(torch.stack(logits_list), t_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataset_pairs):.4f}")

print("\n✅ Treino Finalizado. Testando:")
for p in ["Amor?", "Paz?"]:
    out = model.generate(tokenizer.encode(p), device)
    print(f"{p} -> {tokenizer.decode(out)}")
