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
docs = docs[:-3]



# selected_docs = np.array(docs)[np.where(mask == 1)]
for doc in docs:
    if "<|split|>" in doc.page_content:
        raise ValueError("Error: '<|split|>' found in document content.")

context = " <|split|> ".join([doc.page_content for doc in docs])
# prompt = context + " " + target.page_content

input_ids = tokenizer.encode(context)
original_string = tokenizer.decode(input_ids[1:])

def replace_and_track_indices(context, target):
    indices = []
    while target in context:
        index = context.find(target)
        indices.append(index)
        context = context[:index] + context[index + len(target):]
    return context, indices

# Example usage
target = " <|split|>"
new_context, split_indices = replace_and_track_indices(original_string, target)

print("Number of splits: ", len(split_indices))
print(len(docs))

start_index = 0
for i, doc in enumerate(docs):
    if i == len(split_indices):
        doc.page_content = new_context[start_index:]
    else:
        doc.page_content = new_context[start_index:split_indices[i]]
        start_index = split_indices[i]

for doc in docs:
    print(doc.page_content)


print(len(original_string))
input_ids = tokenizer.encode(original_string)
total_string = ''
doc_index = 0
for input_id in input_ids[1:]:
    total_string += tokenizer.decode(input_id)
    if total_string == docs[doc_index].page_content:
        total_string = ''
        doc_index += 1
        print("Found a document")
print(len(original_string))

# print("Original String: ", original_string[17:])

# print("Original String: ", len(context))