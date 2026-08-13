import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class TokenizerWrapper:
    def __init__(self, encoding_name: str = "gpt2"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        # GPT-2 default vocab size is 50,257
        self.vocab_size = self.tokenizer.n_vocab
        # End-of-Text token (<|endoftext|>) ID is 50,256
        self.eot_token_id = self.tokenizer.eot_token
        self.pad_token_id = self.tokenizer

    def encode(self, text: str, allowed_special: set = {"<|endoftext|>"}) -> list[int]:
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, text: list[int]) -> str:
        return self.tokenizer.decode(text)


class LanguageModelingDataset(Dataset):
    """PyTorch Dataset that handles both single strings and lists of documents,
    splitting them into fixed-length input-target pairs.
    """

    def __init__(self, text, tokenizer: TokenizerWrapper, max_seq_len: int):
        self.max_seq_len = max_seq_len
        encoded_ids = []

        # If you passed a list of strings (e.g., lines from a file)
        if isinstance(text, list):
            for doc in text:
                encoded_ids.extend(tokenizer.encode(doc))
                # Add an End-Of-Text token between separate documents
                encoded_ids.append(tokenizer.eot_token_id)
        # If you passed a single giant string
        else:
            encoded_ids = tokenizer.encode(text)

        self.tokens = torch.tensor(encoded_ids, dtype=torch.long)

    def __len__(self) -> int:
        # Total number of complete sequences we can extract
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx: int):
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len

        # x and y will always be exactly 'max_seq_len' long
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1: end_idx + 1]

        return x, y

# --- Verification & Tensor Shape Test ---
if __name__ == "__main__":
    # Sample corpus
    sample_corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Large language models rely on transformer architectures. "
        "Attention mechanisms allow networks to model long-range dependencies."
    ) * 20  # Duplicate to ensure sufficient length

    # 1. Initialize Tokenizer
    tokenizer = TokenizerWrapper(encoding_name="gpt2")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"End-Of-Text Token ID: {tokenizer.eot_token_id}")

    # 2. Encode and Decode Test
    encoded = tokenizer.encode("Hello, building a GPT model from scratch!")
    decoded = tokenizer.decode(encoded)
    print(f"\nSample Encoding: {encoded}")
    print(f"Sample Decoding: '{decoded}'")

    # 3. PyTorch Dataset & DataLoader Test
    seq_len = 16
    batch_size = 4

    dataset = LanguageModelingDataset(
        text=sample_corpus, tokenizer=tokenizer, max_seq_len=seq_len
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Grab a single batch
    x_batch, y_batch = next(iter(dataloader))

    print("\n--- Tensor Shape Verification ---")
    print(f"Input batch shape (x):  {x_batch.shape}  -> Expected: [{batch_size}, {seq_len}]")
    print(f"Target batch shape (y): {y_batch.shape}  -> Expected: [{batch_size}, {seq_len}]")

    # Verify token shifting
    print("\nToken shifting sanity check (Sample 0):")
    print(f"x[0]: {x_batch[0][:5].tolist()}")
    print(f"y[0]: {y_batch[0][:5].tolist()}")