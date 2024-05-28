import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

import tiktoken

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

# Set the model to evaluation mode
model.eval()

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

# chunks = splitter.split_text(input_text)

# Define the input text
input_texts = [input_text[i:i+4000] for i in range(0, len(input_text), 4000)]

prompt_first_half = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assistant specialized in splitting a corpus into thematically consistent sections. Read the following corpus and identify the points where splits should occur, such to create consecutive strips of similar themes. Respond with the text of each chunk. Here is an example response:

CORPUS: "The Ferari is a fast car. The Lamborghini is also a fast car. Toast can be made with bread. The toaster is used to make toast."

RESPONSE:
First Chunk: "The Ferari is a fast car. The Lamborghini is also a fast car."
Second Chunk: "Toast can be made with bread. The toaster is used to make toast."

Here is the corpus you will be working with:

<|begin_of_corpus|>"""

token_count = 0
defined_chunks = []

for input_text in input_texts:
    promt_second_half = """<|end_of_corpus|>

    Respond with the chunk and it's corresponding text from the corpus above:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    First Chunk: TEXT
    Second Chunk: TEXT
    Third Chunk: TEXT
    """

    len_of_first_half = len(tokenizer.encode(prompt_first_half, return_tensors='pt')[0])
    len_of_second_half = len(tokenizer.encode(promt_second_half, return_tensors='pt')[0])
    len_of_corpus = len(tokenizer.encode(input_text, return_tensors='pt')[0])

    print(len_of_first_half, len_of_second_half, len_of_corpus)

    prompt = prompt_first_half + input_text + promt_second_half

    # Tokenize the input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print(len(input_ids))

    # Get the model's outputs without tracking gradients
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    prompt_len = len(input_ids[-1])

    third_chunk_index = (prompt_len-5, prompt_len-1)
    second_chunk_index = (prompt_len-10, prompt_len-6)
    first_chunk_index = (prompt_len-15, prompt_len-11)

    start_corpus_index = len_of_first_half
    end_of_corpus_index = len_of_first_half + len_of_corpus

    window_size = 300

    chunks = []

    print("got here 1")
    
    torch.cuda.empty_cache()
    
    del model

    

    print("got here 1.5")

    attention_matrix_original = torch.max(torch.stack(outputs.attentions), dim=0).values.detach().numpy()

    print("got here 2")

    del outputs

    for chunk_index in [first_chunk_index, second_chunk_index, third_chunk_index]:
        attention_matrix = attention_matrix_original.copy()

        attention_matrix = attention_matrix[:, -20:, :, :]
        attention_matrix = np.max(attention_matrix, axis=1)

        attention_matrix = attention_matrix[0, chunk_index[0]:chunk_index[1], :]
        attention_matrix = np.sum(attention_matrix, axis=0)

        # No overlap with previous chunks
        for prev_chunk in chunks:
            attention_matrix[prev_chunk[0]:prev_chunk[1]] = -np.inf

        # Initialize the best sum and best position
        best_sum = -np.inf
        best_position = -1

        # Iterate over the possible start positions of the window
        for start_position in range(start_corpus_index, end_of_corpus_index - window_size + 1):
            # Calculate the sum of the window
            window_sum = np.sum(attention_matrix[start_position:start_position + window_size])
            
            # If this sum is better than the current best, update the best sum and best position
            if window_sum > best_sum:
                best_sum = window_sum
                best_position = start_position

        chunks.append((best_position, best_position + window_size))

    defined_chunks.append([(chunk[0]-len_of_first_half+token_count, chunk[1]-len_of_first_half+token_count) for chunk in chunks])
    print(defined_chunks)

    token_count += len_of_corpus


# Unload the model from memory
del model
torch.cuda.empty_cache()
