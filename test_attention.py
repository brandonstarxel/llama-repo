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

print(len(input_ids[-1]))

# Get the model's outputs without tracking gradients
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

# Print all the keys on outputs
print(outputs.keys())

print("Shape of logits:", outputs.logits.shape)


# Print the shape of attentions
print(len(outputs.attentions))
print("Shape of attentions:", outputs.attentions[-1].shape)
# print(outputs.attentions)


