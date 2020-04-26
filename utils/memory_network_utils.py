from transformers import BertTokenizer
from constants import constants
from more_itertools import padded, split_before
import re


def split_most_probable_paragraph(paragraph_text):
    return paragraph_text.split(" ")


def pad_document(self, documents):
    document_split = [re.split("(<P>) | (</P>)", d) for d in documents]
    max_doc_len = max()

    return list(padded(documents, [], max_doc_len))


def split_documents_into_segments(document_text):
    def split_function(x):
        # lookup_set = ["<P>", "<Tr>", "<Table>", "<Ul>", "<Ol>", "<Dl>", "<Li>", "<Dd>", "<Dt>"]
        # lookup_set = ["<P>", "<Table>", "<Ul>", "<Ol>", "<Dl>", "<Li>", "<Dd>", "<Dt>"]
        lookup_set = ["<P>", "<Table>", "<Ul>", "<Ol>", "<Dl>", "<Dd>", "<Dt>"]
        lookup_set = lookup_set + [i.upper() for i in lookup_set]
        return x in set(lookup_set)
    return list(split_before(document_text.split(), lambda x: split_function(x)))


class MemoryNetworkUtils:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(constants.BERT_bu)
        self.bos = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.eos = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def split_most_probable_paragraph(self, paragraph_text):
        return paragraph_text.split(" ")

    def pad_question(self, texts):
        max_lens = max([len(text) for text in texts])
        return [[self.bos] +
                list(padded(text, self.pad, max_lens)) +
                [self.eos] for text in texts]

    def change_offset(self, dt, start, end):
        context = []
        start_diff = 0
        end_diff = 0
        clean = re.compile("<.*?>")
        for i, token in enumerate(dt):
            if re.sub(clean, '', token) != '':
                context.append(token)
            elif i < start:
                start_diff += 1
                end_diff += 1
            elif i < end:
                end_diff += 1
        new_start = start - start_diff
        new_end = end - end_diff
        return context, new_start, new_end
