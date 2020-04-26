from torch.utils.data.dataset import IterableDataset
from utils import split_documents_into_segments
from more_itertools import padded, flatten
from utils import pad_text, pad_document
from transformers import BertTokenizer
from constants import constants
from pprint import pprint
import numpy as np
import torch
import json


tokenizer = BertTokenizer.from_pretrained(constants.BERT_bu)
bos = tokenizer.convert_tokens_to_ids("[CLS]")
eos = tokenizer.convert_tokens_to_ids("[SEP]")
pad = tokenizer.convert_tokens_to_ids("[PAD]")


class NQDatasetForBERT(IterableDataset):
    def __init__(self, file_name):
        self.tokenizer = BertTokenizer.from_pretrained(constants.BERT_bu)
        self.bos = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.eos = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.file_name = file_name

    def __iter__(self):
        with open(self.file_name, "r") as f:
            for line in f:
                data = json.loads(line)
                question = data["question_text"]
                document = data["document_text"]
                annotation = data["annotations"][-1]
                long_answer = annotation.get("long_answer", None)
                if long_answer:
                    long_answer_strt_idx = long_answer["start_token"]
                    long_answer_end_idx = long_answer["end_token"]
                else:
                    long_answer_strt_idx = None
                    long_answer_end_idx = None

                short_answer = annotation.get("short_answer", None)
                if short_answer:
                    short_answer_strt_idx = short_answer["start_token"]
                    short_answer_end_idx = short_answer["end_token"]
                else:
                    short_answer_strt_idx = None
                    short_answer_end_idx = None
                if not long_answer:# and not short_answer:
                    continue

                yield question, \
                      document, \
                      long_answer_strt_idx, \
                      long_answer_end_idx, \
                      short_answer_strt_idx, \
                      short_answer_end_idx


def collate_fn(batch):
    question, \
    document, \
    long_answer_strt_idx, \
    long_answer_end_idx, \
    short_answer_strt_idx, \
    short_answer_end_idx = zip(*batch)

    question = np.array(question)
    document = np.array([[" ".join(split_d) for split_d in split_documents_into_segments(d)] for d in document])
    important_paragraph = -1
    document_indices = []
    document_index_counter = 0
    for i in document:
        # paragraph level
        for j_idx, j in enumerate(i):
            paragraph_indices = []
            # token level
            for _ in j.split():
                paragraph_indices.append(document_index_counter)
                if long_answer_strt_idx[0] == document_index_counter:
                    important_paragraph = j_idx
                document_index_counter += 1
            document_indices.append(paragraph_indices)

    return question,\
           document,\
           document_indices,\
           important_paragraph,\
           long_answer_strt_idx,\
           long_answer_end_idx,\
           short_answer_strt_idx,\
           short_answer_end_idx

    # return question, \
    #        padded_document_segments, \
    #        document_segments_flat,\
    #        long_answer_strt_idx, \
    #        long_answer_end_idx, \
    #        short_answer_strt_idx, \
    #        short_answer_end_idx

