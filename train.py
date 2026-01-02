import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"

embed_dim = 256
block_size = 128
dropout = 0.2

tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "additional_special_tokens": ["<ADLIB>", "<CHORUS>", "<VERSE>", "<SONG>"]
}

tokenizer.add_special_tokens(special_tokens)
tokenizer.model_max_length = 100_000


class CartiDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.tokens = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(
            self.tokens[idx + 1 : idx + self.block_size + 1], dtype=torch.long
        )

        return x, y


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.head_dim = head_dim

        self.Wq = nn.Linear(embed_dim, head_dim, bias=False)  # CxC
        self.Wk = nn.Linear(embed_dim, head_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask)

    def forward(self, x):
        _, T, _ = x.shape

        Q = self.Wq(x)  # TxC @ CxC -> TxC
        K = self.Wk(x)
        V = self.Wv(x)

        weights = (Q @ K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # TxC @ CxT -> TxT / sqrt(C)

        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ V  # TxT @ TxC -> TxC
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0
        head_dim = embed_dim // n_heads

        self.heads = nn.ModuleList(
            [SelfAttentionHead(embed_dim, head_dim) for _ in range(n_heads)]
        )

        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, n_heads):
        super().__init__()
        self.block_size = block_size

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.size()
        assert T <= self.block_size

        tok_emb = self.tok_embed(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(positions)

        x = tok_emb + pos_emb

        x = x + self.attn(self.ln(x))

        logits = self.fc(x)

        loss = None
        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def train(model, dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    dataset = CartiDataset("carti_dataset.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = Transformer(
        vocab_size=len(tokenizer), embed_dim=embed_dim, block_size=block_size, n_heads=4
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train(model, dataloader, optimizer, device, epochs=3)

    torch.save(model.state_dict(), "models/v1.1.pth")

    def generate(model, tokenizer, prompt="<SONG> <VERSE>", max_new_tokens=50):
        model.eval()
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        for _ in range(max_new_tokens):
            logits, _ = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokenizer.decode(tokens[0])

    print(
        "\n\n---------------------------------------\n\n" + generate(model, tokenizer)
    )
