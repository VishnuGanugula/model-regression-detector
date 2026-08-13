import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from transformers import GPT2LMHeadModel
from src.model.transformer import CausalTransformer
from src.model.tokenizer import TokenizerWrapper


def load_openai_weights():
    print("1. Downloading official OpenAI GPT-2 (124M) weights...")
    # This downloads the pre-trained weights into a Hugging Face model
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    print("2. Initializing custom CausalTransformer...")
    tokenizer = TokenizerWrapper()
    # GPT-2 Small configuration
    custom_model = CausalTransformer(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_layers=12,
        max_seq_len=1024
    )
    custom_sd = custom_model.state_dict()

    print("3. Mapping weights (and transposing Conv1D to nn.Linear)...")

    # --- 1. Embeddings ---
    custom_sd["token_emb.weight"].copy_(hf_sd["transformer.wte.weight"])
    custom_sd["pos_emb.weight"].copy_(hf_sd["transformer.wpe.weight"])

    # --- 2. Transformer Blocks ---
    for i in range(12):
        # Layer Norms
        custom_sd[f"blocks.{i}.ln_1.weight"].copy_(hf_sd[f"transformer.h.{i}.ln_1.weight"])
        custom_sd[f"blocks.{i}.ln_1.bias"].copy_(hf_sd[f"transformer.h.{i}.ln_1.bias"])
        custom_sd[f"blocks.{i}.ln_2.weight"].copy_(hf_sd[f"transformer.h.{i}.ln_2.weight"])
        custom_sd[f"blocks.{i}.ln_2.bias"].copy_(hf_sd[f"transformer.h.{i}.ln_2.bias"])

        # Feed Forward Network (Requires .t() transpose)
        custom_sd[f"blocks.{i}.ffwd.net.0.weight"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"].t())
        custom_sd[f"blocks.{i}.ffwd.net.0.bias"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"])
        custom_sd[f"blocks.{i}.ffwd.net.2.weight"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        custom_sd[f"blocks.{i}.ffwd.net.2.bias"].copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"])

        # Attention Mapping
        # HF combines Q, K, V into a single matrix. We must split it into three.
        # Note: If your attention.py uses a single combined nn.Linear(d_model, d_model * 3),
        # you can remove the split and just transpose it.
        c_attn_weight = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"]
        c_attn_bias = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]

        q_w, k_w, v_w = c_attn_weight.split(768, dim=1)
        q_b, k_b, v_b = c_attn_bias.split(768, dim=0)

        # Apply to custom MultiHeadAttention (Assuming W_query, W_key, W_value naming

        # Apply to custom MultiHeadAttention (Using your exact variable names)
        try:
            custom_sd[f"blocks.{i}.attn.w_q.weight"].copy_(q_w.t())
            custom_sd[f"blocks.{i}.attn.w_q.bias"].copy_(q_b)

            custom_sd[f"blocks.{i}.attn.w_k.weight"].copy_(k_w.t())
            custom_sd[f"blocks.{i}.attn.w_k.bias"].copy_(k_b)

            custom_sd[f"blocks.{i}.attn.w_v.weight"].copy_(v_w.t())
            custom_sd[f"blocks.{i}.attn.w_v.bias"].copy_(v_b)

            # Output projection
            custom_sd[f"blocks.{i}.attn.W_o.weight"].copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.weight"].t())
            custom_sd[f"blocks.{i}.attn.W_o.bias"].copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.bias"])

        except KeyError as e:
            print(f"\n[!] KeyError: {e}")
            return None

    # --- 3. Final Layer Norm ---
    custom_sd["ln_f.weight"].copy_(hf_sd["transformer.ln_f.weight"])
    custom_sd["ln_f.bias"].copy_(hf_sd["transformer.ln_f.bias"])

    # --- 4. Output Head ---
    # In GPT-2, the output head weights are tied to the token embedding weights.
    custom_sd["lm_head.weight"].copy_(hf_sd["transformer.wte.weight"])

    print("4. Saving mapped weights to 'custom_gpt2_124M.pth'...")
    torch.save(custom_model.state_dict(), "custom_gpt2_124M.pth")
    print("Success! Your custom model now has the brain of OpenAI's GPT-2.")


if __name__ == "__main__":
    load_openai_weights()