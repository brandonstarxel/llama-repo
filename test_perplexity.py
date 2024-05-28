import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
input_text = "Once upon a time"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Calculate the loss without tracking gradients
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

# Calculate perplexity
perplexity = torch.exp(loss)

print(f'Perplexity: {perplexity.item()}')