import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from color_text import HighlightedTextImage

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:1000]

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get model outputs
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Calculate perplexity for each token
perplexities = torch.exp(loss).view(-1)

# Print the perplexity for each token
tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

def clean_token(token):
    return token.replace('Ġ', ' ').replace('Ċ', '\n')


def perplexity_color(perplexity):
    light_blue = (3, 252, 236)
    dark_blue = (0, 135, 126)

    if perplexity < 1:
        return light_blue
    elif perplexity <= 3:
        # Linearly interpolate between light and dark blue as perplexity increases from 1 to 3
        interp_factor = (perplexity - 1) / 2
        blue_color = tuple(int(light_blue[i] + interp_factor * (dark_blue[i] - light_blue[i])) for i in range(3))
        return blue_color
    else:
        return dark_blue

text_tuples = []

for token, perplexity in zip(tokens[1:], perplexities):
    cleaned_token = clean_token(token)
    print(f'Token: {cleaned_token}, Perplexity: {perplexity.item()}')
    text_tuples.append((cleaned_token, perplexity_color(perplexity.item())))

highlighted_text_image = HighlightedTextImage(text_tuples, font_size=48, width=1600)
highlighted_text_image.generate_image()
highlighted_text_image.save_image('highlighted_text.png')




# import numpy as np
# import matplotlib.pyplot as plt

# # Convert perplexities tensor to numpy array
# perplexities_np = perplexities.numpy()

# # Print statistics
# print(f'Mean perplexity: {np.mean(perplexities_np)}')
# print(f'Min perplexity: {np.min(perplexities_np)}')
# print(f'Max perplexity: {np.max(perplexities_np)}')
# print(f'Median perplexity: {np.median(perplexities_np)}')
# print(f'Standard deviation of perplexity: {np.std(perplexities_np)}')

# # Plot histogram
# plt.figure(figsize=(10, 6))
# plt.hist(perplexities_np, bins=50, color='blue', edgecolor='black')
# plt.title('Histogram of Perplexities')
# plt.xlabel('Perplexity')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig('perplexities_histogram.png')
