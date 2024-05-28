from color_text import HighlightedTextImage
import numpy as np

def calculate_entropy(probabilities):
    # Filter out zero probabilities to avoid log(0) which is undefined
    nonzero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    return entropy

# Example distribution
probabilities = np.array([0.5, 0.25, 0.000000000001])

entropy = calculate_entropy(probabilities)
print(f"Entropy: {entropy} bits")

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:3000]

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
# input_text = "I was walking through a park. Then I saw a"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

print(f"Length of input text: {len(input_text)} characters")
print(f"Length of input_ids: {len(input_ids[0])} tokens")


# Get the model's outputs without tracking gradients
with torch.no_grad():
    outputs = model(input_ids)

# Extract the logits for the next token
# next_token_logits = outputs.logits[:, -1, :]

# Extract the logits for all tokens
all_logits = outputs.logits
print(f"Shape of all_logits: {all_logits.shape}")


# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(all_logits, dim=-1)
print(f"Shape of probabilities: {probabilities.shape}")


# Calculate entropy for each token
entropies = []
for prob in probabilities[0]:
    # Shape (128256,)
    entropy = calculate_entropy(prob.detach().numpy())

    entropies.append(entropy)

def light_blue_heat_map(value):
    """
    Converts a value between 0 and 1 into a light blue heat map color in (r, g, b) format.
    
    Parameters:
    value (float): A float between 0 and 1.
    
    Returns:
    tuple: A tuple representing the (r, g, b) values of the light blue heat map color.
    """
    value = max(0, min(value, 1))
    
    # Define the light blue color range
    start_color = (173, 216, 230)  # Light Blue (lighter end)
    end_color = (0, 0, 255)        # Blue (darker end)

    # Interpolate between the start and end colors
    r = start_color[0] + (end_color[0] - start_color[0]) * value
    g = start_color[1] + (end_color[1] - start_color[1]) * value
    b = start_color[2] + (end_color[2] - start_color[2]) * value
    
    return (int(r), int(g), int(b))

# # Example usage
# value = 0.5
# color = light_blue_heat_map(value)
# print(f"Value: {value} -> Color: {color}")


text_tuples = []

# Print the entropy for each token

def clean_token(token):
    return token.replace('Ġ', ' ').replace('Ċ', '\n')

tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

for token, entropy in zip(tokens, entropies):
    print(f'Token: {clean_token(token)}, Entropy: {entropy}')
    text_tuples.append((clean_token(token), light_blue_heat_map((entropy-1.)/10)))

# for token, perplexity in zip(tokens[1:], perplexities):
#     cleaned_token = clean_token(token)
#     print(f'Token: {cleaned_token}, Perplexity: {perplexity.item()}')
#     text_tuples.append((cleaned_token, perplexity_color(perplexity.item())))

highlighted_text_image = HighlightedTextImage(text_tuples, font_size=48, width=1600)
highlighted_text_image.generate_image()
highlighted_text_image.save_image('entropy_text.png')