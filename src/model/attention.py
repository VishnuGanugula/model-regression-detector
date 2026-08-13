import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
        Computes the scaled dot-product attention scores.
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # d_k is the dimension of a single attention head
        d_k = q.size(-1)

        # 1. Calculate Q * K^T / sqrt(d_k)
        # Transpose only the last two dimensions of K for the dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            # Q * K^T / sqrt(d_k) + M (where M is causual mask matrix)

        atten_weights = torch.softmax(scores, dim=-1)
        atten_weights = self.dropout(atten_weights)

        # (Q * K^T / sqrt(d_k) + M) * V
        output = torch.matmul(atten_weights, v)

        return output, atten_weights

class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention block that splits the input into multiple heads,
        computes attention in parallel, and concatenates the results.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def split_heads(self, x, batch_size):
        """
                Reshapes the tensor to isolate the num_heads dimension.
                Original: (batch_size, seq_len, d_model)
                Split: (batch_size, seq_len, num_heads, d_k)
                Transposed: (batch_size, num_heads, seq_len, d_k)
        """
        return  x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        """
            x shape: (batch_size, seq_len, d_model)
        """

        batch_size = x.size(0)

        # 1. Linear Projections (Creating Queries, Keys, and Values)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Split into multiple heads for parallel computation
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. Adjust mask shape for the multi-head dimensions
        if mask is not None:
            # Mask shape becomes: (batch_size, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # 4. Compute Scaled Dot-Product Attention
        output, attn_weights = self.attention(q, k, v, mask)

        # 5. Re-concatenate the heads
        # Transpose back to: (batch_size, seq_len, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()

        # Flatten the last two dimensions back into d_model
        # Shape becomes: (batch_size, seq_len, d_model)
        output = output.view(batch_size, -1, self.d_model)

        # 6. Final Linear Projection
        output = self.W_o(output)

        return output

    # --- Verification & Tensor Shape Test ---
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 4
    seq_len = 16
    d_model = 512
    num_heads = 8

    # Create dummy input tensor (simulating embedded tokens)
    dummy_x = torch.randn(batch_size, seq_len, d_model)

    # Create a lower-triangular causal mask for autoregressive generation
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

    # Initialize Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # Run the forward pass
    output = mha(dummy_x, mask=causal_mask)

    print("--- Multi-Head Attention Verification ---")
    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {output.shape} -> Expected: [{batch_size}, {seq_len}, {d_model}]")