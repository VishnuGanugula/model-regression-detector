import sys
import yaml
import torch
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import your custom architecture components
from src.model.tokenizer import TokenizerWrapper
from src.model.transformer import CausalTransformer
from src.feature.schemas import EmailClassificationResult, EmailCategory


def load_prompt_config(prompt_version: str = "v1") -> dict:
    project_root = Path(__file__).resolve().parents[2]
    prompt_path = project_root / "prompts" / f"{prompt_version}.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_local_model():
    """Initializes the custom CausalTransformer and loads pre-trained weights."""
    tokenizer = TokenizerWrapper()

    # Initialize your architecture (124M parameters configuration)
    model = CausalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=768,  # GPT-2 Small hidden size
        num_heads=12,  # GPT-2 Small heads
        num_layers=12,  # GPT-2 Small layers
        max_seq_len=1024  # GPT-2 Small context
    )

    project_root = Path(__file__).resolve().parents[2]
    fine_tuned_weights_path = project_root / "fine_tuned_gpt2.pth"
    base_weights_path = project_root / "custom_gpt2_124M.pth"
    weights_path = fine_tuned_weights_path

    if not fine_tuned_weights_path.exists():
        if base_weights_path.exists():
            print(
                f"Weights file {fine_tuned_weights_path.name} not found. "
                f"Using {base_weights_path.name}."
            )
            weights_path = base_weights_path
        else:
            print(
                f"Weights file {fine_tuned_weights_path.name} not found. "
                "Automatically downloading and mapping OpenAI GPT-2 weights..."
            )
            from load_weights import load_openai_weights
            load_openai_weights()
            weights_path = base_weights_path

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("Loading pre-trained weights into the model...")
    model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set to evaluation mode
    return tokenizer, model


# Load model into memory once so it doesn't reload on every function call
TOKENIZER, MODEL = setup_local_model()


def classify_email(email_text: str, prompt_version: str = "v1") -> EmailClassificationResult:
    config = load_prompt_config(prompt_version)

    # 1. Format the few-shot prompt
    full_prompt = config["prompt_template"].replace("{email_text}", email_text.strip())

    # 2. Tokenize input
    input_ids = torch.tensor([TOKENIZER.encode(full_prompt)], dtype=torch.long)

    # 3. Generate exactly ONE new token (the category word)
    # We use low temperature and top_k=1 to make it highly deterministic
    generated_ids = MODEL.generate(
        input_ids,
        max_new_tokens=1,
        temperature=config.get("temperature", 0.1),
        top_k=config.get("top_k", 1)
    )

    # 4. Decode the single generated token
    # We slice out the input prompt and only decode the newly generated token
    new_token_id = generated_ids[0][-1].item()
    output_text = TOKENIZER.decode([new_token_id]).strip().lower()

    # 5. Parse and validate the output against our Pydantic schema
    try:
        # Check if GPT-2 outputted a valid category string
        category = EmailCategory(output_text)
    except ValueError:
        # If GPT-2 hallucinated (e.g., outputted "money" instead of "billing")
        category = EmailCategory.UNKNOWN

    return EmailClassificationResult(category=category)


if __name__ == "__main__":
    sample = "I cannot log into my dashboard, my password reset link is broken."
    result = classify_email(sample, prompt_version="v1")

    print("--- Local GPT-2 Feature Verification ---")
    print(f"Input: {sample}")
    print(f"Extracted Category: {result.category.value}")