from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
# Load your fine-tuned adapter on top
peft_model = PeftModel.from_pretrained(base_model, "./my-finetuned-lora")

# Create a test prompt
prompt = "User: What is the capital of France?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a response with the fine-tuned model
with torch.no_grad():
    outputs = peft_model.generate(**inputs, max_new_tokens=50)
print("Fine-tuned response:", tokenizer.decode(outputs[0], skip_special_tokens=True))