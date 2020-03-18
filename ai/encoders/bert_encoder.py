import torch
from transformers import *
from utils.encoder_utils import pad_text
from .encoder import Encoder


class BERT_Encoder(Encoder):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def embed(self, inputs):
        tokenized_inputs = self.tokenizer.tokenize(inputs)
        padded_inputs = pad_text(tokenized_inputs)
        padded_ids = self.tokenizer.convert_tokens_to_ids(padded_inputs)
