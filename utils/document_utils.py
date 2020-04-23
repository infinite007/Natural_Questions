from more_itertools import padded
import re


def pad_document(documents):
    max_doc_len = max([len(re.split("(<P>) | (</P>)", d)) for d in documents])
    return list(padded(documents, [], max_doc_len))