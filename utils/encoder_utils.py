from more_itertools import padded


def pad_text(text, bos, eos, pad):
    max_seq_len = max([len(t) for t in text])
    return [[bos] + list(padded(t, pad, max_seq_len)) + [eos] for t in text]