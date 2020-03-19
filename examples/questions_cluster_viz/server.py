from flask import Flask, jsonify, request
from collections import Counter, OrderedDict
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from argparse import ArgumentParser
import numpy as np
import json, os

app = Flask(__name__)
CORS(app)

args = ArgumentParser()

args.add_argument("--host",
                  default="0.0.0.0",
                  type=str)

args.add_argument("--port",
                  default=8000,
                  type=int)

args.add_argument("--data_path",
                  required=True,
                  type=str)

args.add_argument("--n_clusters",
                  required=True,
                  type=int)

args.add_argument("--batch_size",
                  required=True,
                  type=int)

args.add_argument("--max_iter",
                  required=True,
                  type=int)

parsed = vars(args.parse_args())
hostname = parsed["host"]
port = parsed["port"]
n_clusters = parsed["n_clusters"]
batch_size = parsed["batch_size"]
max_iters = parsed["max_iter"]

labels_file = os.path.join(parsed["data_path"],
                           "cluster_labels_%d_%d_%d.json" % (n_clusters, batch_size, max_iters))
questions_file = os.path.join(parsed["data_path"], "questions.json")


with open(labels_file, "r") as f:
    cluster_labels = np.array(json.load(f))

with open(questions_file, "r") as f:
    questions = np.array(json.load(f))


@app.route("/", methods=['GET'])
def index():
    return "App is up and running."


@app.route("/questions", methods=['POST'])
def get_questions_in_cluster():
    data = request.get_json()
    cluster_number = int(data["clusterNumber"])
    return jsonify(questions[np.where(cluster_labels == cluster_number)].tolist())


@app.route("/clusterCount", methods=['GET'])
def get_cluster_counts():
    od = OrderedDict()
    for k, v in sorted(Counter(cluster_labels).items(), key=lambda x: x[0]):
        od[str(k)] = v
    return json.dumps(od, indent=4)


if __name__ == '__main__':
    http_server = WSGIServer((hostname, port), app)
    http_server.serve_forever()

