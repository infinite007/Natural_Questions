from ai.models import MemoryNetwork
from ai.dataloaders import NQDatasetForBERT, collate_fn
import torch.nn as nn
from torch.utils.data import DataLoader
from more_itertools import padded
from pprint import pprint
import torch.optim
import re
from tqdm import trange
import numpy as np

epochs = 1
nq_dataset = NQDatasetForBERT("/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/nq-train.jsonl")
data_loader = DataLoader(nq_dataset, batch_size=1, collate_fn=collate_fn)
mem_nn_model = MemoryNetwork()

# print(mem_nn_model.forward(["Hi, There", "How are you?", "Bye Bye"],
#                             ["1 Hi, There", "1 How are you?", "1 Bye Bye",
#                              "2 Hi, There", "2 How are you?", "2 Bye Bye",
#                              "3 Hi, There", "3 How are you?", "3 Bye Bye"],
#                            [["1 Hi, There", "1 How are you?", "1 Bye Bye"],
#                             ["2 Hi, There", "2 How are you?", "2 Bye Bye"],
#                             ["3 Hi, There", "3 How are you?", "3 Bye Bye"]]))


for batch in data_loader:
    q, c, c_flat, la_strt, la_end, sa_strt, sa_end = batch
    print("c flat : ", c_flat)

    print(mem_nn_model.forward(q, c_flat, c))
    break

# for i in trange(epochs):
#     for batch in data_loader:
#         # print(batch)
#         break
#     break
    #     question, \
    #     document, \
    #     long_answer_strt_idx, \
    #     long_answer_end_idx, \
    #     short_answer_strt_idx, \
    #     short_answer_end_idx = zip(*batch)
    #     print([len(re.split("(<P>)|(</P>)", j)) for j in document])
    #     print(document[0].split("</P>"))
    #
    #     break
    # break

    # lstart, lend = mem_nn_model.forward(question, document, document)
    # print(lstart, lend)



