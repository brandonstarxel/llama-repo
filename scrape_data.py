with open('/home/paperspace/llama-repo/output.log', 'r') as file:
    data = file.read()
# print(data)

filtered_data = [line for line in data.split('\n') if line.startswith("['") or line.startswith('["')]

import ast

parsed_data = [ast.literal_eval(line) for line in filtered_data]

total_chunks = []

for chunk in parsed_data:
    total_chunks.extend(chunk)

print(total_chunks)