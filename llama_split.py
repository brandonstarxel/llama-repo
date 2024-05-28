import transformers
import torch
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

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

# input_text = input_text[:2000]
chunks = splitter.split_text(input_text)

def get_prompt(chunked_input):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an assistant specialized in splitting text into thematically consistent sections. "
                "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. "
                "Your response should be in the form: 'split_after: 3, 5'."
            )
        },
        {
            "role": "user", 
            "content": (
                "CHUNKED_TEXT: " + chunked_input + "\n\n"
                "Respond only with the IDs of the chunks where you believe a split should occur."
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

while True:
    if current_chunk >= len(chunks) - 4:
        break

    token_count = 0

    chunked_input = ''

    for i in range(current_chunk, len(chunks)):
        token_count += num_tokens_from_string(chunks[i])
        chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
        if token_count > 800:
            break

    messages = get_prompt(chunked_input)

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

    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', result_string)

    # Convert the found numbers to integers
    numbers = list(map(int, numbers))

    print(numbers)

    split_indices.extend(numbers)

    current_chunk = numbers[-1]

    if len(numbers) == 0:
        break

print(split_indices)