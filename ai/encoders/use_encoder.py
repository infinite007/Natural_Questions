import os
import tensorflow_hub as hub
from .encoder import Encoder
from constants import constants


class USEEncoder(Encoder):
    def __init__(self, trainable=False, version=4):
        super().__init__()
        os.environ["TFHUB_CACHE_DIR"] = constants.pretrained_dir
        self.model_url = constants.USE_url % version
        self.get_embeddings = hub.KerasLayer(self.model_url, trainable=trainable)

    def embed(self, inputs):
        return self.get_embeddings(inputs)
