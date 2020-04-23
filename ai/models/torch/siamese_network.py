import torch
import torch.nn as nn
from ai.encoders import BERTEncoder


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = BERTEncoder()

    def forward(self, query, questions):
        query_embedding = self.embedding_model.embed(query)
        questions_embeddings = self.embedding_model.embed(questions)

        return torch.matmul(query_embedding, torch.transpose(questions_embeddings, 1, 0))
