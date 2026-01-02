import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"

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


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size=128):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        batch_size, seq_length = x.size()
        tok_emb = self.tok_embed(x)

        positions = (
            torch.arange(seq_length, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
        )
        pos_emb = self.pos_embed(positions)

        x = tok_emb + pos_emb
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


dataset = CartiDataset("carti_dataset.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = Transformer(vocab_size=len(tokenizer), embed_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train(model, dataloader, optimizer, device, epochs=3)

torch.save(model.state_dict(), "v1.0.pth")


def generate(model, tokenizer, prompt="<SONG> <VERSE>", max_new_tokens=50):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])


print(generate(model, tokenizer))
