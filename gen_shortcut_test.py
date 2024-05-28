import transformers
import torch
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


prompt = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an assistant specialized in splitting text into thematically consistent sections. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. Your response should be in the form: 'split_after: 3, 5'.<|eot_id|><|start_header_id|>user<|end_header_id|>

CHUNKED_TEXT: CHUNKED_INPUT

Respond only with the IDs of the chunks where you believe a split should occur.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

split_after: 1, 2<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

print(prompt)