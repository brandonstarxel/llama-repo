import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from color_text import HighlightedTextImage

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:1000]

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get model outputs
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits

print(f"Time taken: {end_time - start_time} seconds")
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = input_ids[:, 1:].contiguous()
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# # Calculate perplexity for each token
# perplexities = torch.exp(loss).view(-1)