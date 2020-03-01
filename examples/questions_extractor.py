import json
import os
from tqdm import tqdm

data_path = "/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data"
questions = []
with open(os.path.join(data_path, "nq-train.jsonl"), "r") as f:
    for line in tqdm(f):
        questions.append(json.loads(line)["question_text"])


with open(os.path.join(data_path, "questions.json"), "w") as f:
    json.dump(questions, f)