from flask import Flask, jsonify, request
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from argparse import ArgumentParser
from utils import LSHUtils, get_encoder
from ai.encoders import USEEncoder

app = Flask(__name__)
CORS(app)

args = ArgumentParser()

args.add_argument("--host",
                  default="0.0.0.0",
                  type=str)

args.add_argument("--port",
                  default=8000,
                  type=int)

args.add_argument("--lsh_id",
                  required=True,
                  type=str)

args.add_argument("--encoder",
                  default="use",
                  type=str)

parsed = vars(args.parse_args())
hostname = parsed["host"]
port = parsed["port"]
lsh_id = parsed["lsh_id"]
encoder = get_encoder(parsed["encoder"])()

lsh_utils = LSHUtils(lsh_id)


@app.route("/get_support", methods=["POST"])
def get_support_set():
    data = request.get_json()
    query = data["query"]
    return lsh_utils.get_support_set(query, encoder)


@app.route("/get_support_multiple", methods=["POST"])
def get_support_set():
    data = request.get_json()
    query = data["query"]
    return [lsh_utils.get_support_set(q, encoder) for q in query]
