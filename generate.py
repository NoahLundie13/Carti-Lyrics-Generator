import torch
from transformers import AutoTokenizer

from train import Transformer

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "additional_special_tokens": ["<ADLIB>", "<CHORUS>", "<VERSE>", "<SONG>"]
}

tokenizer.add_special_tokens(special_tokens)
tokenizer.model_max_length = 100_000

model = Transformer(
    vocab_size=len(tokenizer),
    embed_dim=256,
    block_size=128,
    n_heads=4,
).to(device)

state_dict = torch.load("models/v1.1.pth", map_location=device)
model.load_state_dict(state_dict)


def generate(model, tokenizer, prompt="<SONG> <VERSE>", max_new_tokens=120):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])


print(
    "\n\n---------------------------------------\n\n"
    + generate(model, tokenizer)
    + "\n\n---------------------------------------\n\n"
)
