import torch
import torch.nn.functional as F
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

import torch
import torch.nn.functional as F

def calculate_loss_for_segment(input_ids, model, start, end):
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, start:end] = 1

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(input_ids.size(0), -1)

    relevant_loss = loss[:, start:end-1]
    avg_loss = relevant_loss.mean()
    return avg_loss.item()

def find_optimal_segmentation(input_text, model, tokenizer, max_segments):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    n = input_ids.size(1)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    segment_starts = [-1] * (n + 1)

    for j in range(1, n + 1):
        for i in range(j):
            if j - i > max_segments:
                continue
            loss = calculate_loss_for_segment(input_ids, model, i, j)
            if dp[i] + loss < dp[j]:
                dp[j] = dp[i] + loss
                segment_starts[j] = i

    # Reconstruct the segmentation
    segments = []
    end = n
    while end > 0:
        start = segment_starts[end]
        segments.append((start, end))
        end = start

    segments.reverse()
    return segments

# Example usage
max_segments = 10  # Maximum length of each segment
optimal_segments = find_optimal_segmentation(input_text, model, tokenizer, max_segments)

print("Optimal Segments (start, end):")
for seg in optimal_segments:
    print(seg)
