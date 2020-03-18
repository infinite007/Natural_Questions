# from transformers import pipeline
# nlp = pipeline('sentiment-analysis')
# print(nlp('We are very happy to include pipeline into the transformers repository.'))


import torch
from transformers import *
import numpy as np

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Encode text
print(tokenizer.encode("i like eggs", add_special_tokens=True))
print(tokenizer.tokenize("i like eggs", add_special_tokens=True))
print(tokenizer.encode("my car is totally damaged", add_special_tokens=True))
print(tokenizer.tokenize("my car is totally damaged", add_special_tokens=True))
print("tokenizing list of texts")
print(tokenizer.tokenize("[CLS] [SEP]", add_special_tokens=True))
print(tokenizer.batch_encode_plus(["Here is the sentence I want embeddings for.",
                                   "[CLS] [SEP]", "[PAD]"], add_special_tokens=True))

text = "Here is the sentence I want embeddings for."
marked_text = text

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)
encoded_text = tokenizer.encode(marked_text)
from utils.encoder_utils import pad_text
print("tokenized text to Ids : ")
print(tokenizer.convert_tokens_to_ids(pad_text(tokenized_text)))

# Print out the tokens.
print (tokenized_text)
print (encoded_text)

input_ids = torch.tensor([tokenizer.encode("king", add_special_tokens=True)])
# last_hidden_states = torch.mean(model(input_ids)[0][0], dim=0)
last_hidden_states = model(input_ids)[0][0]
print(last_hidden_states.shape)

input_ids_2 = torch.tensor([tokenizer.encode("queen", add_special_tokens=True)])
# last_hidden_states_2 = torch.mean(model(input_ids_2)[0][0], dim=0)
last_hidden_states_2 = model(input_ids_2)[0][0]
print(last_hidden_states_2.shape)
lhs = last_hidden_states.detach().numpy()
lhs_2 = last_hidden_states_2.detach().numpy()
# print(lhs/np.linalg.norm(lhs))
# print(lhs_2/np.linalg.norm(lhs_2))
print(np.dot(lhs / np.linalg.norm(lhs), lhs_2.T / np.linalg.norm(lhs_2)))
# print(last_hidden_states.detach().numpy())
