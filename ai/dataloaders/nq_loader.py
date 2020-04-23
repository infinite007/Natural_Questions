from torch.utils.data.dataset import IterableDataset
from transformers import BertTokenizer
from utils import pad_text, pad_document
from more_itertools import padded, flatten
from constants import constants
import torch
from pprint import pprint
from utils import split_documents_into_segments
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
                    long_answer_end_idx = long_answer["start_token"]
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
                if not long_answer and not short_answer:
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

    # batched_questions = [["CLS"] + q.split() + ["SEP"] for q in question]

    document_segments = [split_documents_into_segments(d) for d in document]
    max_len = max([len(b) for b in document_segments])
    padded_document_segments = [[["<CLS>"] + list(padded(d, "<PAD>", max_len)) + ["<SEP>"] for d in ds] for ds in document_segments]
    document_segments_flat = [k for i in padded_document_segments for j in i for k in j]

    # return batched_questions,\
    #        padded_document_segments,\
    #        long_answer_strt_idx,\
    #        long_answer_end_idx,\
    #        short_answer_strt_idx,\
    #        short_answer_end_idx

    return question, \
           padded_document_segments, \
           document_segments_flat,\
           long_answer_strt_idx, \
           long_answer_end_idx, \
           short_answer_strt_idx, \
           short_answer_end_idx

