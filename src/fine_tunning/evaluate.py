from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
adapter_path = Path(__file__).resolve().parents[2] / "my-finetuned-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
# Load your fine-tuned adapter on top
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
peft_model.eval()


def generate_response(prompt: str, max_new_tokens: int = 50) -> str:
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


# Create a test prompt
prompt = "What is GPxtsxLmp"

# Same model with the adapter switched off, for a before/after comparison
with peft_model.disable_adapter():
    print("Base response:", generate_response(prompt))

print("Fine-tuned response:", generate_response(prompt))
