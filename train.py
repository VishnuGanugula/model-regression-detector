import torch
import torch.nn.functional as F
from torch.optim import AdamW # FIXED: Proper class import
from torch.utils.data import DataLoader

# FIXED: Removed the invalid "from model import tokenizer" line
from src.model.tokenizer import TokenizerWrapper, LanguageModelingDataset
from src.model.transformer import CausalTransformer

# HyperParameters
vocab_size = 50257
max_seq_len = 128
d_model = 256
num_heads = 8
num_layers = 4
batch_size = 8
learning_rate = 3e-4
epochs = 5

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# --- 3. Prepare Data ---
raw_text = "The quick brown fox jumps over the lazy dog. " * 500

tokenizer = TokenizerWrapper()
dataset = LanguageModelingDataset(text=raw_text, tokenizer=tokenizer, max_seq_len=max_seq_len)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# FIXED: Turned shuffle off for validation
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = CausalTransformer(vocab_size, d_model, num_heads, num_layers, max_seq_len).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0

    for x_val, y_val in dataloader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        logits = model(x_val)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1))
        total_loss += loss.item()

    model.train()
    return total_loss / len(dataloader)

model.train()
for epoch in range(epochs):
    total_train_loss = 0.0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Calculate Cross-Entropy Loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        loss.backward()

        # Update model weights
        optimizer.step()

        total_train_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

    # Run Validation at the end of each epoch
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = evaluate_loss(model, val_loader)

    print("-" * 50)
    print(f"End of Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
    print("-" * 50)