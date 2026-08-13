import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward neural network applied to each token independently.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Standard practice is to expand the hidden dimension by 4x
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # GELU is standard for GPT models
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder block utilizing Pre-LayerNorm architecture.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Residual dropout after attention, before adding to the shortcut connection
        self.drop_1 = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout)

    def forward(self, x, mask=None):
        # Shortcut connection 1: Attention
        x = x + self.drop_1(self.attn(self.ln_1(x), mask))

        # Shortcut connection 2: Feed-Forward
        x = x + self.ffwd(self.ln_2(x))
        return x


class CausalTransformer(nn.Module):
    """
    The complete GPT-style Causal Language Model.
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Embeddings: Tokens + Positions
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # The stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm and Output Head (Logits)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying: Share weights between token embeddings and the output head
        self.token_emb.weight = self.lm_head.weight

        # Register the causal mask to the model device
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, idx):
        """
        idx shape: (batch_size, seq_len) containing integer token IDs
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Create positional indices: [0, 1, 2, ... seq_len-1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        # Add Token Embeddings and Positional Embeddings
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        # Crop the causal mask to the current sequence length
        mask = self.causal_mask[:, :seq_len, :seq_len]

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization and prediction head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generates new tokens by repeatedly passing the sequence
        through the model and sampling from the predicted probability distribution.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the maximum sequence length
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Get the logits for the current sequence
            logits = self(idx_cond)

            # Pluck out the logits at the final position and scale by temperature
            logits = logits[:, -1, :] / temperature

            # Optional: Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --- Full End-to-End Generation Test ---
if __name__ == "__main__":
    from tokenizer import TokenizerWrapper

    # 1. Initialize Tokenizer and Model
    tokenizer = TokenizerWrapper()
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=256
    )

    # 2. Prepare the starting prompt
    prompt = "Hello, building a GPT model from scratch"
    print(f"Starting prompt: '{prompt}'")

    # Encode prompt and convert to PyTorch tensor with batch dimension: (1, seq_len)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    # 3. Generate new tokens
    print("\nGenerating untrained text...")

    # Generate 20 new tokens
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=10
    )

    # 4. Decode the result back to text
    output_text = tokenizer.decode(generated_ids[0].tolist())

    print("\n--- Final Output ---")
    print(output_text)