import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
input_text = "I was walking through a park. Then I saw a"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the model's outputs without tracking gradients
with torch.no_grad():
    outputs = model(input_ids)

# Extract the logits for the next token
next_token_logits = outputs.logits[:, -1, :]

# Convert logits to probabilities
probabilities = torch.softmax(next_token_logits, dim=-1)

# Get the top 10 tokens and their probabilities
top_k = 10
top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

# Convert token indices to tokens
top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices[0]]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(top_k_tokens, top_k_probs[0].tolist())
plt.xlabel('Tokens')
plt.ylabel('Probability')
plt.title('Top 10 Token Probabilities')
plt.savefig('top_10_token_probabilities.png')
