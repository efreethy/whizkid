import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model name (Hugging Face ID)
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token="secret")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    token="secret"
)

# Define the prompt
prompt = "You are Ferbie, a friendly toy who loves talking with children. Child: Hi Ferbie, how are you today?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate tokens **one by one** (streaming output)
stream = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7, 
    streamer=True  # Enables token streaming
)

# Print tokens as they are generated
print("\nFerbie:", end="", flush=True)

for token in stream:
    text = tokenizer.decode(token, skip_special_tokens=True)
    print(text, end="", flush=True)

print("\n")  # New line at the end
