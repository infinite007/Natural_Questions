import os


class Constants:
    # paths
    root_dir = os.path.dirname(__file__)
    ai_dir = os.path.join(root_dir, "ai")
    encoder_dir = os.path.join(ai_dir, "encoders")
    data_dir = os.path.join(root_dir, "data")
    lsh_dir = os.path.join(data_dir, "LSH")
    questions_dir = os.path.join(data_dir, "questions.json")
    nq_train_dir = os.path.join(data_dir, "nq-train.jsonl")
    deploy_dir = os.path.join(root_dir, "deploy")
    preprocessing_dir = os.path.join(root_dir, "preprocessing")
    pretrained_dir = os.path.join(root_dir, "pretrained")
    # encoder urls
    USE_url = "https://tfhub.dev/google/universal-sentence-encoder/%d"
    BERT_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16"
    BERT_bu = "bert-base-uncased"


constants = Constants()
