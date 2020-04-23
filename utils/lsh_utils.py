import json, os
import numpy as np
from constants import constants


def create_hash_string(output_vectors):
    hash_values_multiple = [(output_vector > 0).squeeze().tolist() for output_vector in output_vectors]
    return ["".join(['1' if i else '0' for i in hash_values]) for hash_values in hash_values_multiple]


def hamming_distance(v1, v2):
    assert len(v1) == len(v2), "length of the first vector is not equal to length of the second vector"
    return sum(i == j for i, j in zip(v1, v2)) / len(v1)


def get_best_bin(query_hash, bin_keys):
    min_ham_dist = min([(bk, hamming_distance(query_hash, bk))
                        for bk in bin_keys], key=lambda x: x[1])
    return min_ham_dist[0]


class LSHUtils:
    def __init__(self, lsh_id):
        with open(os.path.join(constants.lsh_dir, lsh_id, "bins.json")) as f:
            self.bins = json.load(f)
        self.lsh_planes = np.load(os.path.join(constants.lsh_dir, lsh_id, "lsh_planes.npy"))

    def get_support_set(self, query, encoder):
        query_embedding = encoder.embed([query])

        lsh_shape = self.lsh_planes.shape
        output = np.matmul(query_embedding,
                           self.lsh_planes.reshape(-1, lsh_shape[-1]).transpose(1, 0))
        output = output.reshape(lsh_shape[0], -1)
        hash_strings = create_hash_string(output)
        support_set = set()
        for bin, hash_string in zip(self.bins, hash_strings):
            support_set.update(bin[hash_string])
        return list(support_set)

