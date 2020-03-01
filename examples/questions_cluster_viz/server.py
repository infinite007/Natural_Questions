from flask import Flask, jsonify, request
from collections import Counter, OrderedDict
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import numpy as np
import json

app = Flask(__name__)
CORS(app)


with open("../../data/cluster_labels.json", "r") as f:
    cluster_labels = np.array(json.load(f))

with open("../../data/questions.json", "r") as f:
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
    http_server = WSGIServer(('0.0.0.0', 9000), app)
    http_server.serve_forever()

