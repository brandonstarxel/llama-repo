import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define a batch of input texts
input_texts = ["I was walking through a park. Then I saw a",
               "The weather today is sunny and warm. It makes me feel",
               "Artificial intelligence is transforming the world in"]

# Set pad_token as eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_texts = input_texts*100

# Tokenize the input texts
input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)

# Get the model's outputs without tracking gradients
with torch.no_grad():
    outputs = model(**input_ids)

# Extract the logits for all tokens
all_logits = outputs.logits

# all_logits is a tensor of shape (batch_size, sequence_length, vocab_size)
print(all_logits.shape)
