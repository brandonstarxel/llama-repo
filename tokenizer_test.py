import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Read the input text
with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:3000]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')
print(f"Length of input text: {len(input_text)} characters")

# Convert the token IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Print the length of the tokens
print(f"Length of tokens: {len(tokens)}")

# Map tokens back to the original text
token_to_text_map = []
current_position = 0

for token in tokens:
    # Decode token to get its text representation
    token_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(token), skip_special_tokens=True)
    token_length = len(token_text)
    
    # Find the corresponding text in the original input
    for i in range(current_position, len(input_text)):
        if input_text[i:i + token_length] == token_text:
            token_to_text_map.append((token, input_text[i:i + token_length], i, i + token_length))
            current_position = i + token_length
            break

# Print the mapping from tokens to original text segments
for token, text_segment, start_idx, end_idx in token_to_text_map:
    print(f"Token: {token}, Text Segment: '{text_segment}', Start Index: {start_idx}, End Index: {end_idx}")
