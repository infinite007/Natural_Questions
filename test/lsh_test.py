from utils import LSHUtils, get_encoder
from pprint import pprint
import numpy as np
from argparse import ArgumentParser

parameters = ArgumentParser()

parameters.add_argument("--encoder",
                        type=str,
                        default="bert")

parameters.add_argument("--id",
                        type=str,
                        required=True)

parameters.add_argument("--debug",
                        type=str,
                        required=False)

args = vars(parameters.parse_args())

encoder_model = get_encoder(args["encoder"])
lsh_id = args["id"]
debug = args["debug"]

lsh_utils = LSHUtils(lsh_id)
encoder = encoder_model()

query = input("QUERY>>>")
output = lsh_utils.get_support_set(query, encoder)
# pprint(output)
print(len(output))
if debug:
    test_element = input("TEST_ELEMENT>>>")
    pprint([(idx, i) for idx, i in enumerate(output) if test_element in i])

output_2 = np.matmul(encoder.embed([query]), np.array(encoder.embed(output)).T)
best_out = np.argmax(output_2)
if debug:
    print(output_2.shape)
    print(best_out)
    print(output_2[:, best_out])
    print(output_2[:, int(input("INDEX>>>"))])
print(output[best_out])
