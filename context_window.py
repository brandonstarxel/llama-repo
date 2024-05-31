import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the model to evaluation mode
model.eval()

# Define the input text
input_text = "I was walking through a park. Then I saw a"

with open('/home/paperspace/llama-repo/corpuses/wikitexts.md', 'r') as file:
    input_text = file.read()

input_text = input_text[:4000]

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

print(docs[1])

docs = docs[3:]

def assign_doc_indexes(docs):
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
    return input_ids

# samples = 20
# X = []
# y = []

# for i, doc in enumerate(docs):
#     X.append(np.zeros(20, len(docs)))
#     y.append(np.zeros(20))

for i, doc in enumerate(docs):
    doc.metadata['X'] = None
    doc.metadata['y'] = None
    doc.metadata['index'] = i


def get_prob_for_mask(docs, mask):
    selected_docs = np.array(docs)[np.where(mask == 1)]
    input_ids = assign_doc_indexes(selected_docs)

    with torch.no_grad():
        outputs = model(input_ids)

    token_logits = outputs.logits[:, :-1, :]
    softmax = torch.nn.Softmax(dim=-1)
    token_probs = softmax(token_logits)

    for doc in selected_docs:
        if doc.metadata['index'] == 0:
            continue

        X_i = mask[:doc.metadata['index']]
        if doc.metadata['X'] is None:
            doc.metadata['X'] = X_i
        else:
            doc.metadata['X'] = np.vstack((doc.metadata['X'], X_i))

        target_tokens_len = doc.metadata['end_index'] - doc.metadata['start_index']
        target_token_ids = input_ids[0][doc.metadata['start_index']-1:doc.metadata['end_index']-1]
        
        target_sentence_prob = 1.0
        for i, token_id in enumerate(target_token_ids):
            target_sentence_prob *= token_probs[0, i, token_id].item()
        print("Log-Probability Norm of target sentence: ", np.log(target_sentence_prob)/target_tokens_len)

        y_i = np.log(target_sentence_prob)/target_tokens_len
        if doc.metadata['y'] is None:
            doc.metadata['y'] = np.array([y_i])
        else:
            doc.metadata['y'] = np.append(doc.metadata['y'], y_i)

    # return np.log(target_sentence_prob)    

# def get_prob_for_mask(docs, mask, target):
#     selected_docs = np.array(docs)[np.where(mask == 1)]
#     context = " ".join([doc.page_content for doc in selected_docs])
#     prompt = context + " " + target.page_content

#     input_ids = tokenizer.encode(prompt, return_tensors='pt')

#     with torch.no_grad():
#         outputs = model(input_ids)

#     target_tokens_len = len(tokenizer.encode(" " + target.page_content))

#     # Decode the target tokens to check if they have been correctly encoded
#     decoded_tokens = tokenizer.decode(input_ids[0][-target_tokens_len+1:])
#     # print("Decoded Tokens: ", decoded_tokens)

#     target_token_logits = outputs.logits[:, -target_tokens_len:-1, :]

#     # print(target_token_logits.shape)
#     # torch.Size([1, 19, 128256])

#     softmax = torch.nn.Softmax(dim=-1)
#     target_token_ids = input_ids[0][-target_tokens_len+1:]
#     target_token_probs = softmax(target_token_logits)
#     target_sentence_prob = 1.0
#     for i, token_id in enumerate(target_token_ids):
#         target_sentence_prob *= target_token_probs[0, i, token_id].item()
#     print("Log-Probability of target sentence: ", np.log(target_sentence_prob))
#     return np.log(target_sentence_prob)    


# # Get the model's outputs without tracking gradients
# start_time = time.time()
# with torch.no_grad():
#     outputs = model(input_ids)
# end_time = time.time()
# print(f"Model Output Time: {end_time - start_time} seconds")
# # Extract the logits for the next token
# next_token_logits = outputs.logits[:, -1, :]

unique_masks = set()
while len(unique_masks) < 60:
    mask = np.random.choice([0, 1], size=len(docs))
    mask_tuple = tuple(mask)
    if mask_tuple not in unique_masks:
        unique_masks.add(mask_tuple)
        get_prob_for_mask(docs, mask)

# print(docs)

correlation_mat = np.zeros((len(docs), len(docs)))

print("Length of docs: ", len(docs))

for i, doc in enumerate(docs):
    if doc.metadata['y'] is None or len(doc.metadata['y']) < 2:
        continue

    X = doc.metadata['X']
    y = doc.metadata['y']

    print(f"Index of doc: {i}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Lasso model
    # lasso = Lasso(alpha=0.1)
    lasso = Lasso(alpha=0.1)

    # Fit the model to the training data
    lasso.fit(X_train, y_train)

    # Predict the output for the test set
    y_pred = lasso.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Get the coefficients from the model
    coefficients = lasso.coef_

    # Print the non-zero coefficients and their corresponding feature indices
    important_features = np.where(coefficients != 0)[0]

    print(f"Important Features: {important_features}")
    print(f"Coefficients: {coefficients[important_features]}")

    for j, coef in enumerate(coefficients):
        correlation_mat[i, j] = coef


# Symmetrize the correlation matrix
correlation_mat = (correlation_mat + correlation_mat.T)

import matplotlib.pyplot as plt

# Plot the correlation matrix
plt.figure(figsize=(10,10))
plt.imshow(correlation_mat, cmap='cool', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')



# import matplotlib.pyplot as plt

# # Plot the actual vs predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')

# # Plot a line for perfect fit
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# plt.savefig('actual_vs_predicted.png')
