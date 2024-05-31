import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:2000]

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tokenizer.encode(string)
    num_tokens = len(encoding)
    return num_tokens

splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            length_function=num_tokens_from_string,
            add_start_index=True
        )

docs = splitter.create_documents([input_text])

input_ids = None
start_index = 0
for doc in docs:
    page_string = doc.page_content+' '
    current_input_ids = tokenizer.encode(page_string, return_tensors='pt')
    doc.metadata['start_index'] = start_index
    if input_ids is None:
        doc.metadata['start_index'] = 1
        start_index = 1
        input_ids = current_input_ids
    else:
        if current_input_ids[0][0] == 128000:
            current_input_ids = current_input_ids[:,1:]
        input_ids = torch.cat((input_ids, current_input_ids[:,:]), dim=1)
    if start_index != 1:
        start_index += len(current_input_ids[0])
    else:
        start_index += len(current_input_ids[0])-1
    doc.metadata['end_index'] = start_index-1 

print(input_ids)

print(' '.join([tokenizer.decode(input_ids[0][doc.metadata['start_index']:doc.metadata['end_index']]) for doc in docs]))