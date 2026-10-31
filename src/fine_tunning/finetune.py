from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

project_root = Path(__file__).resolve().parents[2]

# 1. Load a small model and tokenizer (e.g., SmolLM2-135M)
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # Rank of the update matrices
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.05,
    target_modules="all-linear", # Adapt every linear layer for better memorization
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Should show a very small percentage

# 3. Load and tokenize your dataset
dataset = load_dataset(
    "json",
    data_files=str(Path(__file__).with_name("train.jsonl")),
    split="train",
)

def format_and_tokenize(example):
    # Convert the messages into a single string prompt
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(text, truncation=True, max_length=256)

tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# 4. Set up training arguments
training_args = TrainingArguments(
    output_dir=str(project_root / "results"),
    num_train_epochs=30,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none", # Disable cloud logging
    use_cpu=True,     # Explicitly force CPU usage
)

# 5. Create a Trainer and start training
def causal_lm_collator(features):
    # Mask only padding tokens in the labels. DataCollatorForLanguageModeling masks
    # labels equal to pad_token_id, and this tokenizer uses the EOS token as pad,
    # which would remove all EOS supervision and the model would never stop.
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=causal_lm_collator,
)

trainer.train()
trainer.save_model(str(project_root / "my-finetuned-lora")) # Saves only the small LoRA adapter
