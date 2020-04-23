import json
from constants import constants


def get_questions_from_data():
    with open(constants.questions_dir, "r") as f:
        return json.load(f)
