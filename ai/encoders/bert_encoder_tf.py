import tensorflow as tf
import tensorflow_hub as hub
from utils.encoder_utils import pad_text
from .encoder import Encoder
from constants import constants


class BERTEncoder2(Encoder):
    def __init__(self, last_layer_only=True,
                 top_k_layers=4,
                 no_grad=True,
                 model_dir=constants.pretrained_dir):
        super().__init__()
        self.top_k_layers = top_k_layers
        self.model = hub.Module(constants.BERT_url)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bos = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.eos = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.pad = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def embed(self, inputs):
        tokenized_inputs = self.tokenizer.batch_encode_plus(inputs)["input_ids"]
        padded_ids = torch.tensor(pad_text(tokenized_inputs, self.bos, self.eos, self.pad))
        if self.no_grad:
            with torch.no_grad():
                _, _, outputs = self.model(padded_ids)
        else:
            _, _, outputs = self.model(padded_ids)
        # length of outputs gives number of outputs in the current model.
        if self.last_layer_only:
            return torch.mean(outputs[-1], dim=1)
        else:
            num_layers = len(outputs) - self.top_k_layers
            top_k_layers_average = sum(outputs[num_layers:]) * (1 / num_layers)
            return torch.mean(top_k_layers_average, dim=1)


