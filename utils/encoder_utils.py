from more_itertools import padded


def pad_text(text, pad_sym="[PAD]"):
    max_seq_len = max([len(t) for t in text])
    return [["[PAD]"] + list(padded(t, pad_sym, max_seq_len)) + ["[SEP]"] for t in text]