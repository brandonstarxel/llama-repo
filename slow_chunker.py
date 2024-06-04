from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

import tiktoken

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            length_function=num_tokens_from_string
        )

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

print(num_tokens_from_string(input_text))

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# def find_invalid_tokens(corpus):
#     invalid_tokens = []
    
#     for text in corpus:
#         tokens = tokenizer.tokenize(text)
#         for token in tokens:
#             if token == tokenizer.unk_token:  # Identify unknown tokens
#                 invalid_tokens.append((token, text))
    
#     return invalid_tokens

# Example corpus
# corpus = [
#     input_text
# ]

# invalid_tokens = find_invalid_tokens(corpus)

# # Output the results
# for token, text in invalid_tokens:
#     print(f"Invalid token '{token}' found in text: '{text}'")


# # Encode and then decode the input_text
# encoded_text = tokenizer.encode(input_text)
# decoded_text = tokenizer.decode(encoded_text[1:])

# print("Decoded Text:", decoded_text[:500])
# print(input_text == decoded_text)

# # Check if the decoded text equals the original input_text
# if decoded_text == input_text:
#     print("The decoded text is equal to the original input text.")
# else:
#     print("The decoded text is not equal to the original input text.")


# input_text = input_text[:2000]
# chunks = splitter.split_text(input_text)

def get_prompt(chunked_input):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an assistant specialized in splitting text into thematically consistent sections. "
                "You must copy the provided text EXACTLY and include <|split|> tags where you believe a split should occur. "
                "Here is an example,\n\n"
                "Corpus: The Ferrari is a fast car. The Lamborghini is also a fast car. The Porsche is a fast car. Apples are good for your health. Try to eat many."
                "\n\nResponse: The Ferrari is a fast car. The Lamborghini is also a fast car. The Porsche is a fast car. <|split|> Apples are good for your health. Try to eat many."
            )
        },
        {
            "role": "user", 
            "content": (
                "Corpus: " + chunked_input + "\n\n"
                "Respond with an exact copy of the corpus with <|split|> tags where you believe a split should occur."
            )
        },
        {
            "role": "assistant", 
            "content": (
                "Response: "
            )
        },
    ]
    return messages

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

import re

current_chunk = 0

split_indices = []

chunks = [input_text[i:i+4000] for i in range(0, len(input_text), 4000)]

for chunk in chunks:

    messages = get_prompt(chunk)

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])

    result_string = outputs[0]["generated_text"][len(prompt):]


print(split_indices)