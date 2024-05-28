import transformers
import torch
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

import tiktoken

# Count the number of tokens in each page_content
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

chunked_input = "This is a test. <|start_chunk_1|> This is another test. <|end_chunk_1|> This is a third test. <|start_chunk_2|> This is a fourth test. <|end_chunk_2|> This is a fifth test."

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

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

print(prompt)

del pipeline

def generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.6, top_p=0.9, terminators=None):
    # Tokenize the initial prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    
    # Store generated tokens
    generated_ids = input_ids
    
    # Generation loop
    for _ in range(max_new_tokens):
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(generated_ids)
        
        logits = outputs.logits[:, -1, :]  # Get logits of the last token

        # Apply temperature scaling
        logits = logits / temperature
        
        # Apply top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('Inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # Append the new token to the generated sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # Check if the generated token is a terminator
        if terminators is not None and next_token_id.item() in terminators:
            break
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


# Example usage
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.6, top_p=0.9, terminators=terminators)
print("Generated text:", generated_text)
