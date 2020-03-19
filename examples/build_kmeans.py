import os, json
import _pickle as pkl
from tqdm import trange
import tensorflow as tf
import tensorflow_hub as hub
from argparse import ArgumentParser
from sklearn.cluster import MiniBatchKMeans

args = ArgumentParser()
args.add_argument("--n_clusters",
                  default=3000,
                  type=int)

args.add_argument("--batch_size",
                  default=10000,
                  type=int)

args.add_argument("--random_state",
                  default=0,
                  type=int)

args.add_argument("--max_iter",
                  default=100,
                  type=int)

args.add_argument("--data_path",
                  default="/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/",
                  type=str)

args.add_argument("--questions_file",
                  default="questions.json",
                  type=str)

args.add_argument("--embedding_model",
                  default="https://tfhub.dev/google/universal-sentence-encoder/4",
                  type=str)


args.add_argument("--model_output_path",
                  default="./",
                  type=str)

parsed = vars(args.parse_args())

os.environ["TFHUB_CACHE_DIR"] = parsed["data_path"]

data_path = parsed["data_path"]
question_file = parsed["questions_file"]
n_clusters = parsed["n_clusters"]
random_state = parsed["random_state"]
batch_size = parsed["batch_size"]
max_iter = parsed["max_iter"]
model_output_path = parsed["model_output_path"]
embedding_model = parsed["embedding_model"]

question_path = os.path.join(data_path, question_file)

with open(question_path, "r") as f:
    questions = json.load(f)


model = MiniBatchKMeans(n_clusters=n_clusters,
                        random_state=random_state,
                        batch_size=batch_size,
                        max_iter=max_iter)

cluster_labels = []
embed = hub.load(embedding_model)
for q in trange(len(questions) // batch_size + 1):
    question_embeddings = embed(questions[q * batch_size:(q + 1) * batch_size])
    cluster_labels.extend(model.fit_predict(question_embeddings))


labels_path = os.path.join(data_path, "cluster_labels_%d_%d_%d.json" % (n_clusters,
                                                                        batch_size,
                                                                        max_iter))

with open(labels_path, "w") as f:
    json.dump([int(i) for i in cluster_labels], f, indent=4)

model_output_file = os.path.join(model_output_path,
                                 "kmeans_%d_%d_%d.pkl" % (n_clusters,
                                                          batch_size,
                                                          max_iter))
with open(model_output_file, "wb") as f:
    pkl.dump(model, f)

