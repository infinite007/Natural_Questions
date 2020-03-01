import tensorflow as tf
import tensorflow_hub as hub
import json, os, csv
from tqdm import tqdm, trange
import numpy as np
from sklearn.manifold import TSNE
import _pickle as pkl
from sklearn.cluster import MiniBatchKMeans

os.environ["TFHUB_CACHE_DIR"] = "../data"

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

data_path = "/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data"

with open(os.path.join(data_path, "questions.json"), "r") as f:
    questions = json.load(f)
#
# if not os.path.exists("question_embeddings.tsv"):
#     with open("question_embeddings.tsv", "ab") as f:
#         for q in trange(len(questions) // 10000 + 1):
#             question_embeddings = embed(questions[q * 10000:(q + 1) * 10000])
#             np.savetxt(f, question_embeddings, delimiter="\t")
#             f.write(b"\n")
#
# kmeans = MiniBatchKMeans(n_clusters=3000,
#                          random_state=0,
#                          batch_size=10000,
#                          max_iter=100)
#
# cluster_labels = []
# for q in trange(len(questions) // 10000 + 1):
#     question_embeddings = embed(questions[q * 10000:(q + 1) * 10000])
#     cluster_labels.extend(kmeans.fit_predict(question_embeddings))
#
# with open("kmeans_3000_10000.pkl", "wb") as f:
#     pkl.dump(kmeans, f)


with open("kmeans_3000_10000.pkl", "rb") as f:
    kmeans = pkl.load(f)

cluster_labels = []
for q in trange(len(questions) // 10000 + 1):
    question_embeddings = embed(questions[q * 10000:(q + 1) * 10000])
    cluster_labels.extend(kmeans.predict(question_embeddings).tolist())


with open(os.path.join(data_path, "cluster_labels.json"), "w") as f:
    json.dump(cluster_labels, f)



