import os
import tensorflow_hub as hub
from .encoder import Encoder


class USE_Encoder(Encoder):
    def __init__(self, version=4):
        super().__init__()
        os.environ["TFHUB_CACHE_DIR"] = "../../data/"
        self.model_url = "https://tfhub.dev/google/universal-sentence-encoder/%d" % version
        self.get_embeddings = hub.load(self.model_url)

    def embed(self, inputs):
        return self.get_embeddings(inputs)