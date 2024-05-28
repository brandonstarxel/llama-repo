from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

import tiktoken

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()


splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            length_function=num_tokens_from_string
        )

chunks = splitter.split_text(input_text)

import random

# # Randomly sample 100 chunks along with their index
# sampled_ids = random.sample(range(len(chunks) - 1), 30)

# input_texts = [chunks[i] + ' ' + chunks[i + 1] for i in sampled_ids]

# Randomly sample 100 pairs of chunks along with their index
sampled_ids = [(random.randint(0, len(chunks) - 1), random.randint(0, len(chunks) - 1)) for _ in range(30)]

input_texts = [chunks[i] + ' ' + chunks[j] for i, j in sampled_ids]

print(f"Number of input texts: {len(input_texts)}")
print(input_texts[0])

import numpy as np

def calculate_entropy(probabilities):
    # Filter out zero probabilities to avoid log(0) which is undefined
    nonzero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    return entropy





import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Set pad_token as eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**input_ids)

all_logits = outputs.logits

# Convert logits to probabilities
probabilities = torch.nn.functional.softmax(all_logits, dim=-1)

mean_entropy = []
for prob_set in probabilities:
    # Convert tensor to numpy and calculate entropy for each probability set
    entropies = np.apply_along_axis(calculate_entropy, 1, prob_set.detach().numpy())
    mean_entropy.append(np.mean(entropies))

plt.hist(mean_entropy, bins=50, color='blue', edgecolor='black')
plt.title('Histogram of Mean Entropies')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.savefig("entropy_histogram_disconnected.png")


print(f'Mean entropy: {np.mean(mean_entropy)}')
print(f'Median entropy: {np.median(mean_entropy)}')
print(f'Min entropy: {np.min(mean_entropy)}')
print(f'Max entropy: {np.max(mean_entropy)}')
print(f'Standard deviation of entropy: {np.std(mean_entropy)}')


# Mean entropy: 4.559882164001465
# Median entropy: 4.1161041259765625
# Min entropy: 2.6347498893737793
# Max entropy: 9.609971046447754
# Standard deviation of entropy: 1.652889609336853


# Mean entropy: 4.99921178817749
# Median entropy: 4.638291358947754
# Min entropy: 2.768834114074707
# Max entropy: 8.819791793823242
# Standard deviation of entropy: 1.4625897407531738