import torch
import torch.nn as nn
from ai.encoders import BERTEncoder
from torch.nn import functional as F
from utils import split_most_probable_paragraph, pad_text
from .feed_forward_network import FeedForwardNetwork
from more_itertools import flatten


class MemoryNetwork:
    def __init__(self, num_hops=3):
        self.encoder = BERTEncoder()
        self.paragraph_query_hidden = FeedForwardNetwork(768, 768, 2, F.relu)
        self.paragraph_query_output = FeedForwardNetwork(768, 768, 1, F.relu)
        self.words_query_hidden = FeedForwardNetwork(768, 768, 2, F.relu)
        self.words_query_output = FeedForwardNetwork(768, 768, 1, F.softmax)
        self.num_hops = num_hops

    def forward(self, queries, paragraph_memories, paragraph_texts):
        queries = self.encoder.embed(queries)
        print(queries.shape)
        batch_size = queries.size(0)
        embedding_size = queries.size(-1)

        queries = queries.reshape(batch_size, 1, embedding_size)

        paragraph_memories = self.encoder.embed(paragraph_memories)
        print(paragraph_memories.shape)

        paragraph_memories = paragraph_memories.reshape(batch_size, -1, embedding_size)

        print(queries.shape, paragraph_memories.shape)

        query_paragraph_similarity = torch.matmul(queries, paragraph_memories.transpose(2, 1))

        query_paragraph_att = F.softmax(query_paragraph_similarity)
        print("query paragraph attention : ", query_paragraph_att)
        print("query paragraph att shape : ", query_paragraph_att.shape)
        important_paragraphs = torch.matmul(query_paragraph_att, paragraph_memories)
        queries = queries + important_paragraphs
        print("important paragraohs : ", important_paragraphs.shape)
        print(important_paragraphs)

        paragraph_argmax = torch.argmax(query_paragraph_att, dim=-1).squeeze().tolist()

        print("paragraph argmax : ", paragraph_argmax)

        probable_paragraph_tokens = pad_text([j[i].split() for i, j in zip(paragraph_argmax, paragraph_texts)], "<CLS>", "<SEP>", "<PAD>")
        probable_paragraph_tokens = list(flatten(probable_paragraph_tokens))
        # probable_paragraph_tokens = split_most_probable_paragraph(most_probable_paragraph)
        probable_paragraph_embeddings = self.encoder.embed(probable_paragraph_tokens)
        print(probable_paragraph_embeddings.shape)
        probable_paragraph_embeddings = probable_paragraph_embeddings.reshape(batch_size,
                                                                              -1,
                                                                              embedding_size)
        print(probable_paragraph_embeddings.shape)

        queries = self.paragraph_query_hidden(queries)
        queries = self.paragraph_query_output(queries)

        important_subparagraphs = torch.matmul(queries,
                                               probable_paragraph_embeddings.transpose(2, 1))
        start = F.softmax(important_subparagraphs)
        print("start shape : ", start.shape)
        start_index = torch.argmax(start, dim=-1)
        print("start index : ", start_index)
        start_masker = torch.arange(0, start.size(-1)) > start_index
        print(start_masker)

        important_subparagraphs_start_index = torch.matmul(start, probable_paragraph_embeddings)
        important_subparagraphs_start_index_att = F.softmax(important_subparagraphs_start_index)

        queries = queries + important_subparagraphs_start_index_att

        queries = self.words_query_hidden(queries)
        queries = self.words_query_output(queries)

        start_masker = start_masker.unsqueeze(-1)

        print("start masker : ", start_masker.shape)
        print("probable paragraph embeddings : ", probable_paragraph_embeddings.shape)

        end = F.softmax(torch.matmul(queries,
                                     (start_masker * probable_paragraph_embeddings).transpose(2, 1)))

        return start, end
