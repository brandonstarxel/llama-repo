from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define the input text
input_text = "Once upon a time"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode the generated token IDs to text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the output
print(output_text)
