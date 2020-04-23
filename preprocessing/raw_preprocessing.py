from utils import tokenize_text
from constants import constants
import json


def data_reader():
    with open(constants.data_dir, "r") as f:
        for line in f:
            yield line


