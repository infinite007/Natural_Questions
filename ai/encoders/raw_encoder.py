import torch
import torch.nn as nn
import torch.nn.functional as F


class RAWEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.embedding_matrix = nn.Embedding(self.params["vocab_size"],
                                             self.params["embedding_size"])
        self.pe = self.positional_encoding()
        self.att_k = nn.Linear(self.params["embedding_size"],
                               int(self.params["embedding_size"] / self.params["num_att_heads"]))
        self.att_q = nn.Linear(self.params["embedding_size"],
                               int(self.params["embedding_size"] / self.params["num_att_heads"]))

    def positional_encoding(self):
        seq_range = torch.range(1, self.params["max_seq_len"]).reshape(-1, 1)
        pos_range = torch.range(1, self.params["embedding_size"])
        pe_matrix = torch.zeros((self.params["max_seq_len"], self.params["embedding_size"]))
        pe_matrix[:, :] = seq_range / torch.pow(10000, 2 * pos_range / self.params["embedding_size"])
        pe_matrix[:, ::2] = torch.sin(pe_matrix[:, ::2])
        pe_matrix[:, 1::2] = torch.cos(pe_matrix[:, 1::2])
        return pe_matrix

    def forward(self, inputs, input_type):
        embeddings = self.embedding_matrix(inputs)
        embeddings = embeddings + self.pe[:embeddings.size(0)]

        if input_type == "query":
            inputs = self.att_q(embeddings)
        return inputs


