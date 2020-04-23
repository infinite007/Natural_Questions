import os
import json
import numpy as np
from tqdm import tqdm
from ai.encoders import *
from constants import constants
from more_itertools import chunked
from argparse import ArgumentParser
from collections import defaultdict
from utils import get_questions_from_data

parameters = ArgumentParser()
parameters.add_argument("--num_planes",
                        type=int,
                        default=15)

parameters.add_argument("--num_bins",
                        type=int,
                        default=15)

parameters.add_argument("--encoder",
                        type=str,
                        default="bert")

parameters.add_argument("--embedding_dimension",
                        type=int,
                        default=768)

parameters.add_argument("--batch_size",
                        type=int,
                        default=1000)

parameters.add_argument("--id",
                        type=str,
                        required=True)

args = vars(parameters.parse_args())

encoder_dict = {
    "bert": BERTEncoder,
    "use": USEEncoder
}

num_planes = args["num_planes"]
num_bins = args["num_bins"]
encoder_model = encoder_dict[args["encoder"]]
embedding_dimension = args["embedding_dimension"]
batch_size = args["batch_size"]
lsh_id = args["id"]

lsh_planes_multiple = np.random.uniform(size=(num_bins, num_planes, embedding_dimension))
encoder = encoder_model()
bins = [defaultdict(list) for _ in range(num_bins)]


for idx, lsh_planes in enumerate(lsh_planes_multiple):
    for questions in tqdm(chunked(get_questions_from_data(), batch_size)):
        questions_embeddings = np.array(encoder.embed(list(questions)))
        outputs = np.matmul(questions_embeddings, lsh_planes.T).squeeze()
        for question, output in zip(questions, outputs):
            hash_str = "".join(['1' if i else '0' for i in (output > 0).tolist()])
            bins[idx][hash_str].append(question)

for bin in bins:
    for k, v in bin.items():
        print(k, " : ", len(v))

save_dir = os.path.join(constants.lsh_dir, lsh_id)
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "bins.json"), "w") as bins_file:
    json.dump([dict(bin) for bin in bins], bins_file, indent=2)

np.save(os.path.join(save_dir, "lsh_planes"), lsh_planes_multiple)

