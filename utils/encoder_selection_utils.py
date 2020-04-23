from ai.encoders import *


def get_encoder(encoder_name):
    return {
        "bert": BERTEncoder,
        "use": USEEncoder
    }[encoder_name]