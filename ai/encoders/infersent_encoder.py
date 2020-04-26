from .models import InferSent
from .encoder import Encoder
from constants import constants
import torch
import torch.nn as nn
import torch.nn.functional as F


class INFERSENTEncoder(Encoder, InferSent):
    def __init__(self, embedding_size=4096, compressed_output_size=None, activation_f=F.tanh, config=None, **kwargs):
        version = kwargs.get("version", None)
        batch_size = kwargs.get("batch_size", None)
        vocab_size = kwargs.get("vocab_size", 100000)
        tokenize_inputs = kwargs.get("tokenize_inputs", True)
        if not config:
            config = {'bsize': batch_size or 1, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                      'pool_type': 'max', 'dpout_model': 0.0, 'version': version or 1}
        Encoder.__init__(self)
        InferSent.__init__(self, config)
        self.tokenize_inputs = tokenize_inputs
        self.load_state_dict(torch.load(constants.infersent_model_dir))
        self.set_w2v_path(constants.infersent_w2v_dir)
        self.build_vocab_k_words(K=vocab_size)
        self.activation_f = activation_f
        self.compressed_output_size = compressed_output_size
        if self.compressed_output_size:
            self.compressed_output = nn.Linear(embedding_size, compressed_output_size)

    def embed(self, inputs):
        if self.compressed_output_size:
            encoder_outputs = torch.from_numpy(self.encode(inputs, tokenize=self.tokenize_inputs))
            compressed_outputs = self.compressed_output(encoder_outputs)
            return self.activation_f(compressed_outputs)
        else:
            return torch.from_numpy(self.encode(inputs, tokenize=self.tokenize_inputs))

