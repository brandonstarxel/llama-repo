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

input_texts = [chunks[i] + ' || ' + chunks[j] for i, j in sampled_ids]

print(input_texts[0])