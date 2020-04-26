import torch
import torch.nn as nn
from ai.encoders import BERTEncoder, INFERSENTEncoder
from torch.nn import functional as F
from utils import split_most_probable_paragraph, pad_text
from .feed_forward_network import FeedForwardNetwork
from .highway_networks import HIGHWAYNetworks
from .mutihead_highway_attention import MULTI_HEAD_HIGHWAY_ATTENTION_Networks
from more_itertools import flatten


class MemoryNetwork(nn.Module):
    def __init__(self, embedding_dimension=4096,
                 compressed_output_size=100, num_att_heads=16,
                 num_paragraph_query_hidden=2, num_paragraph_query_output=1,
                 num_words_query_hidden=2, num_words_query_output=1, num_hops=3):
        super().__init__()
        self.encoder = INFERSENTEncoder(embedding_dimension, compressed_output_size)
        self.num_hops = num_hops
        if compressed_output_size:
            self.multihead_highway_att_q = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(compressed_output_size,
                                                                                 num_att_heads)

            self.multihead_highway_att_k = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(compressed_output_size,
                                                                                 num_att_heads)

            self.multihead_highway_att_v = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(compressed_output_size,
                                                                                 num_att_heads)

            self.paragraph_query_hidden = FeedForwardNetwork(compressed_output_size,
                                                             compressed_output_size,
                                                             num_paragraph_query_hidden, F.relu)

            self.paragraph_query_output = FeedForwardNetwork(compressed_output_size,
                                                             compressed_output_size,
                                                             num_paragraph_query_output, F.relu)

            self.words_query_hidden = FeedForwardNetwork(compressed_output_size,
                                                         compressed_output_size,
                                                         num_words_query_hidden, F.relu)

            self.words_query_output = FeedForwardNetwork(compressed_output_size,
                                                         compressed_output_size,
                                                         num_words_query_output, F.softmax)
        else:
            self.multihead_highway_att_q = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(embedding_dimension,
                                                                                 num_att_heads)

            self.multihead_highway_att_k = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(embedding_dimension,
                                                                                 num_att_heads)

            self.multihead_highway_att_v = MULTI_HEAD_HIGHWAY_ATTENTION_Networks(embedding_dimension,
                                                                                 num_att_heads)

            self.paragraph_query_hidden = FeedForwardNetwork(embedding_dimension,
                                                             embedding_dimension,
                                                             num_paragraph_query_hidden, F.relu)

            self.paragraph_query_output = FeedForwardNetwork(embedding_dimension,
                                                             embedding_dimension,
                                                             num_paragraph_query_output, F.relu)

            self.words_query_hidden = FeedForwardNetwork(embedding_dimension,
                                                         embedding_dimension,
                                                         num_words_query_hidden, F.relu)

            self.words_query_output = FeedForwardNetwork(embedding_dimension,
                                                         embedding_dimension,
                                                         num_words_query_output, F.softmax)

    def forward(self, queries, paragraph_memories, paragraph_texts):
        batch_size = queries.shape[0]
        queries = queries.reshape(-1)
        queries = self.encoder.embed(queries)
        # print("type of queries : ", type(queries))
        embedding_size = queries.shape[-1]
        # print(queries.shape)

        queries = queries.reshape(batch_size, 1, embedding_size)
        queries = self.multihead_highway_att_q(queries)
        # print("highway q : ", queries)
        # print("highway q shape : ", queries.shape)

        paragraph_memories = paragraph_memories.reshape(-1)
        # print("paragraph memories : ", paragraph_memories)
        # print("paragraph memories shape : ", paragraph_memories.shape)

        paragraph_memories = self.encoder.embed(paragraph_memories)
        # print(paragraph_memories.shape)

        paragraph_memories = paragraph_memories.reshape(batch_size, -1, embedding_size)
        paragraph_memories = self.multihead_highway_att_k(paragraph_memories)
        # print("highway k : ", paragraph_memories)
        # print("highway k shape : ", paragraph_memories.shape)

        # print(queries.shape, paragraph_memories.shape)

        query_paragraph_similarity = torch.matmul(queries, paragraph_memories.transpose(2, 1))
        # print("query paragraph similarity : ", query_paragraph_similarity)

        query_paragraph_att = F.softmax(query_paragraph_similarity, dim=-1)
        # print("query paragraph attention : ", query_paragraph_att)
        # print("query paragraph att shape : ", query_paragraph_att.shape)
        important_paragraphs = torch.matmul(query_paragraph_att, paragraph_memories)
        queries = queries + important_paragraphs
        # print("important paragraohs : ", important_paragraphs.shape)
        # print(important_paragraphs)

        paragraph_argmax = torch.argmax(query_paragraph_att, dim=-1).squeeze().tolist()

        # print("paragraph argmax : ", paragraph_argmax)
        # print("paragraph texts : ", paragraph_texts.shape)

        # getting paragraph words here :

        probable_paragraph_tokens = paragraph_texts[0, paragraph_argmax].split()
        # print("probable paragraph tokens : ", probable_paragraph_tokens)
        # print("probable paragraph tokens len : ", len(probable_paragraph_tokens))
        # probable_paragraph_tokens = list(flatten(probable_paragraph_tokens))
        # probable_paragraph_tokens = split_most_probable_paragraph(most_probable_paragraph)
        probable_paragraph_embeddings = self.encoder.embed(probable_paragraph_tokens)
        # print(probable_paragraph_embeddings.shape)
        probable_paragraph_embeddings = probable_paragraph_embeddings.reshape(batch_size,
                                                                              -1,
                                                                              embedding_size)
        probable_paragraph_embeddings = self.multihead_highway_att_v(probable_paragraph_embeddings)
        # print("probable paragraph embeddings shape : ", probable_paragraph_embeddings.shape)

        queries = self.paragraph_query_hidden(queries)
        queries = self.paragraph_query_output(queries)

        important_subparagraphs = torch.matmul(queries,
                                               probable_paragraph_embeddings.transpose(2, 1))
        # print("important subparagraphs : ", important_subparagraphs)
        start = F.softmax(important_subparagraphs, dim=-1)
        # print("start shape : ", start.shape)
        start_index = torch.argmax(start, dim=-1)
        # print("start index : ", start_index)
        start_masker = torch.arange(0, start.size(-1)) > start_index
        # print(start_masker)
        start_masker = start_masker.unsqueeze(-1)
        # print("start masker : ", start_masker.shape)

        important_subparagraphs_start_index = torch.matmul(start, probable_paragraph_embeddings)
        # print("important subparagraphs start index shape : ", important_subparagraphs_start_index.shape)
        important_subparagraphs_start_index_att = F.softmax(important_subparagraphs_start_index, dim=-1)

        queries = queries + important_subparagraphs_start_index_att

        queries = self.words_query_hidden(queries)
        queries = self.words_query_output(queries)

        # print("probable paragraph embeddings : ", probable_paragraph_embeddings.shape)

        start_masked_probable_paragraph_embeddings = start_masker * probable_paragraph_embeddings

        end = F.softmax(torch.matmul(queries,
                                     start_masked_probable_paragraph_embeddings.transpose(2, 1)), dim=-1)

        # long_answer_start = torch.argmax(start, dim=-1).item()
        # end_index = torch.argmax(end, dim=-1)
        # print(start_masker.shape)
        # end_masker = torch.arange(0, end.size(-1)) < end_index

        # end_masker = end_masker.unsqueeze(-1)

        # end_masker = end_masker * start_masker
        # print("end masker : ", end_masker)
        # print("end masker shape : ", end_masker.shape)
        # end_masked_probable_paragraph_embeddings = end_masker * probable_paragraph_embeddings

        # end_similarity_with_query = torch.matmul(end, end_masked_probable_paragraph_embeddings)

        # end_attention_with_query = F.softmax(end_similarity_with_query, dim=-1)

        # queries = queries + end_attention_with_query

        # queries = self.words_query_hidden(queries)
        # queries = self.words_query_output(queries)

        # important_words = torch.matmul(queries, end_masked_probable_paragraph_embeddings.transpose(2, 1))

        # short_answer_start = F.softmax(important_words, dim=-1)
        # short_answer_start_index = torch.argmax(short_answer_start, dim=-1)

        # short_answer_start_masker = torch.arange(0, short_answer_start.size(-1)) > short_answer_start_index
        # print(short_answer_start_masker)
        # short_answer_start_masker = short_answer_start_masker.unsqueeze(-1)
        # print("start masker : ", short_answer_start_masker.shape)

        # important_short_answer_start_index = torch.matmul(short_answer_start, end_masked_probable_paragraph_embeddings)
        # print("important subparagraphs start index shape : ", important_short_answer_start_index.shape)
        # important_short_answer_start_index_att = F.softmax(important_short_answer_start_index, dim=-1)

        #

        #
        # end_masked_probable_paragraph_embeddings = end_masker * start_masked_probable_paragraph_embeddings

        return start.squeeze(), end.squeeze(), query_paragraph_att.squeeze()
