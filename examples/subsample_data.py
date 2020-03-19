import numpy as np
import json
from tqdm import trange, tqdm
from random import sample
from argparse import ArgumentParser
import os

args = ArgumentParser()
args.add_argument("--ratio", default=.3, type=float)
args.add_argument("--output_file", required=True, type=str)
args.add_argument("--labels_file", required=True, type=str)
args.add_argument("--questions_file", required=True, type=str)

parsed = vars(args.parse_args())
labels_file = parsed["labels_file"]
questions_file = parsed["questions_file"]
output_file = parsed["output_file"]
ratio = parsed["ratio"]

with open(labels_file, "r") as f:
    cluster_labels = np.array(json.load(f))

with open(questions_file, "r") as f:
    questions = np.array(json.load(f))

n_clusters = max(cluster_labels)

line_numbers_of_new_samples = []

for i in trange(n_clusters):
    samples_in_each_cluster = np.where(cluster_labels == i)[0].tolist()
    line_numbers_of_new_samples.extend(sample(samples_in_each_cluster,
                                              int(len(samples_in_each_cluster) * ratio)))

with open(output_file, "w") as f:
    json.dump(line_numbers_of_new_samples, f)


line_numbers_of_new_samples = set(line_numbers_of_new_samples)
large_data = "/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/nq-train.jsonl"
small_data = "/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/"
with open(large_data, "r") as train_file,\
        open(os.path.join(small_data, "small_data.json"), "w") as small_file:
    for idx, line in tqdm(enumerate(train_file)):
        if idx in line_numbers_of_new_samples:
            curr_line = json.loads(line)
            json.dump(curr_line, small_file)
            small_file.write("\n")



