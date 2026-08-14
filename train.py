import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from src.model.tokenizer import TokenizerWrapper
from src.model.transformer import CausalTransformer


# --- 1. JSONL Support Ticket Dataset Loader ---
class SupportTicketDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Format text for causal language modeling
                text = f"Email: {item['email']}\nCategory: {item['category']}<|endoftext|>"

                # Tokenize
                tokens = self.tokenizer.encode(text)

                # Truncate or Pad
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                else:
                    tokens = tokens + [0] * (max_seq_len - len(tokens))  # Pad with 0

                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)  # Inputs
        y = torch.tensor(tokens[1:], dtype=torch.long)  # Targets (shifted by 1)
        return x, y


# --- 2. HyperParameters & Setup ---
vocab_size = 50257
max_seq_len = 1024
d_model = 768  # GPT-2 Small hidden size matching your weights
num_heads = 12  # GPT-2 Small heads
num_layers = 12  # GPT-2 Small layers
batch_size = 4  # Small batch size to fit in local memory
learning_rate = 5e-5  # Fine-tuning requires a smaller learning rate
epochs = 2

# Device setup
device = "cpu"
print(f"Training on device: {device}")

# --- 3. Prepare Data (Fixed Path Resolution) ---
project_root = Path(__file__).resolve().parent
data_path = project_root / "data" / "training_data.jsonl"
weights_path = project_root / "custom_gpt2_124M.pth"
output_weights_path = project_root / "fine_tuned_gpt2.pth"

tokenizer = TokenizerWrapper()
dataset = SupportTicketDataset(data_path, tokenizer, max_seq_len=max_seq_len)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- 4. Initialize Model & Load Base Weights ---
model = CausalTransformer(vocab_size, d_model, num_heads, num_layers, max_seq_len).to(device)

if not weights_path.exists():
    print("Pre-trained weights (custom_gpt2_124M.pth) not found! Running load_weights.py...")
    from load_weights import load_openai_weights
    load_openai_weights()

print("Loading pre-trained GPT-2 base weights...")
model.load_state_dict(torch.load(weights_path, map_location=device))

optimizer = AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0

    for x_val, y_val in dataloader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        logits = model(x_val)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1), ignore_index=0)
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)


# --- 5. Training Loop ---
model.train()
print("Starting fine-tuning...")

for epoch in range(epochs):
    total_train_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Calculate Cross-Entropy Loss (ignoring padding index 0)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)

        # Backward pass & update
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 25 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

    # Validation phase
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = evaluate_loss(model, val_loader)

    print("-" * 50)
    print(f"End of Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
    print("-" * 50)

# Save the newly fine-tuned weights
torch.save(model.state_dict(), output_weights_path)
print(f"✅ Fine-tuning complete! Saved fine-tuned weights to: {output_weights_path}")