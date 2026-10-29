# finetune.py
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load a small model and tokenizer (e.g., SmolLM2-135M)
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                # Rank of the update matrices
    lora_alpha=16,      # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], # Common target modules for attention
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
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

# 4. Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none", # Disable cloud logging
    use_cpu=True,     # Explicitly force CPU usage
)

# 5. Create a Trainer and start training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./my-finetuned-lora") # Saves only the small LoRA adapter