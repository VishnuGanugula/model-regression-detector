import json
import yaml
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"

class EmailClassification(BaseModel):
    category: EmailCategory = Field(description="The exact category the email belongs to.")
    summary: str = Field(description="A concise, one-sentence summary.")

def load_prompt(version: str) -> dict:
    """Loads the specified prompt version from the prompts directory."""
    with open(f"prompts/{version}.yaml", "r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# LOCAL MODEL SETUP: No API Keys Required!
# We load a small (0.5B parameter) model that runs easily on a laptop.
# The first time this runs, it will download the weights to your disk.
# ---------------------------------------------------------
print("Loading local model to memory (this may take a minute on first run)...")
local_llm = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", # A great, tiny instruction-tuned model
    device_map="auto" # Automatically uses GPU if available, otherwise CPU
)

def classify_email(email_text: str, prompt_version: str = "v1") -> dict:
    prompt_config = load_prompt(prompt_version)
    
    # Construct the exact text we send to the local model
    system_instruction = prompt_config["system_prompt"]
    full_prompt = f"<|system|>\n{system_instruction}\n<|user|>\n{email_text}\n<|assistant|>\n```json\n"
    
    # Run the model locally
    output = local_llm(
        full_prompt,
        max_new_tokens=150,
        temperature=prompt_config.get("temperature", 0.1),
        return_full_text=False # Only return what the model generated
    )
    
    raw_text = output[0]['generated_text']
    
    # --- Clean and Validate the Local Output ---
    try:
        # Strip out markdown block formatting the model might add
        json_str = raw_text.replace("```json", "").replace("```", "").strip()
        parsed_dict = json.loads(json_str)
        
        # Force the output through our Pydantic guardrails
        validated_data = EmailClassification(**parsed_dict)
        return validated_data.model_dump()
        
    except (json.JSONDecodeError, ValidationError) as e:
        # Local models hallucinate more often than GPT-4. 
        # This catch block is crucial for our CI/CD pipeline to catch regressions!
        print(f"Failed to parse or validate LLM output. Raw output was:\n{raw_text}")
        return {"category": "error", "summary": "Model failed to output valid JSON schema."}

# --- Quick Local Test ---
if __name__ == "__main__":
    test_email = "Hi, I was charged twice for my subscription this month. Can you refund the extra charge?"
    print(f"\nTesting email: '{test_email}'")
    
    result = classify_email(test_email, "v1")
    print("\nClassification Result:")
    print(result)