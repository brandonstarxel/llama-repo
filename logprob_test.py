import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

context = "I was walking through a park."
# target = "Then I saw a dog."
target = "The Ferari is a fast car."

prompt = context + " " + target

input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Print the shape of input_ids for debugging
print("Shape of input_ids: ", input_ids.shape)

with torch.no_grad():
    outputs = model(input_ids)

target_tokens_len = len(tokenizer.encode(" " + target))

# Print the length of target tokens for debugging
print("Length of target tokens: ", target_tokens_len)

# Decode the target tokens to check if they have been correctly encoded
decoded_tokens = tokenizer.decode(input_ids[0][-target_tokens_len+1:])
print("Decoded Tokens: ", decoded_tokens)

target_token_logits = outputs.logits[:, -target_tokens_len:-1, :]

# Print the shape of target_token_logits for debugging
print("Shape of target_token_logits: ", target_token_logits.shape)

softmax = torch.nn.Softmax(dim=-1)
target_token_ids = input_ids[0][-target_tokens_len+1:]

# Print the shape of target_token_ids for debugging
print("Shape of target_token_ids: ", target_token_ids.shape)

target_token_probs = softmax(target_token_logits)
target_sentence_prob = 1.0
for i, token_id in enumerate(target_token_ids):
    print("Token ID: ", token_id)
    print("Token Probability: ", target_token_probs[0, i, token_id].item())
    target_sentence_prob *= target_token_probs[0, i, token_id].item()
print("Probability of target sentence: ", target_sentence_prob)

import numpy as np

logit_of_prob = np.log(target_sentence_prob)
print("Logit of target sentence probability: ", logit_of_prob)
