import re
from ai.models import MemoryNetwork
from ai.dataloaders import NQDatasetForBERT, collate_fn
import torch.nn as nn
from torch.utils.data import DataLoader
from more_itertools import padded
from pprint import pprint
import torch.optim as optim
from tqdm import trange, tqdm
import numpy as np
from argparse import ArgumentParser
import torch



args = ArgumentParser()
args.add_argument("--num_epochs", default=2, type=int)
args.add_argument("--batch_size", default=1, type=int)
args.add_argument("--lr", default=1e-3, type=float)
args.add_argument("--use_cuda", default=True, type=bool)
args.add_argument("--train_data_dir", required=True, type=str)


epochs = 1
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nq_dataset = NQDatasetForBERT("/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/small_data.json")
data_loader = DataLoader(nq_dataset, batch_size=1, collate_fn=collate_fn)
model = MemoryNetwork(compressed_output_size=50, num_att_heads=10)


# for dl_idx, (q, c, c_indices, c_important, la_strt, la_end, sa_strt, sa_end) in enumerate(data_loader):
#     if dl_idx in {1, 6, 19}:
#         start, end, qp_att = model.forward(q, c, c)
#         paragraph = c[0, torch.argmax(qp_att).item()].split()
#         print("question : ", q)
#         print("predicted long answer : ", paragraph)
#     if dl_idx >18:
#         break

    # print(c_indices)
    # print(c_important)
    # # print("c flat : ", c)
    #
    # # print(start.shape)
    # # print(end.shape)
    # break

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
cum_loss = 0
num_steps = 0

# data_loader_tqdm_obj = tqdm(data_loader)

for i in range(epochs):
    for q, c, c_indices, c_important, la_strt, la_end, sa_strt, sa_end in data_loader:
        optimizer.zero_grad()
        lstart, lend, important_paragraph = model.forward(q, c, c)
        print("la start : ", la_strt[0])
        print("la end : ", la_end[0])
        print("c indices : ", c_indices[c_important])
        print("question : ", q)
        print("c : ", c[0][c_important].split())
        print("c : ", c[0][c_important+1].split())

        try:
            modified_la_strt = c_indices[c_important].index(la_strt[0])
        except:
            modified_la_strt = -1
        print("modified la start : ", modified_la_strt)
        try:
            modified_la_end = c_indices[c_important].index(la_end[0])
        except:
            modified_la_end = -1
        print("modified la end : ", modified_la_end)

        if modified_la_end == -1:
            if modified_la_strt == -1:
                modified_la_end = 0
            else:
                modified_la_end = len(c_indices[c_important]) - 1
        if modified_la_strt == -1:
            modified_la_strt = 0

        print("modified la start : ", modified_la_strt)
        print("modified la end : ", modified_la_end)

        # loss_1 = criterion(lstart, torch.tensor(modified_la_strt))
        # loss_2 = criterion(lend, torch.tensor(modified_la_end))
        # loss_3 = criterion(important_paragraph, torch.tensor(c_important))
        # loss_4 = criterion(None, torch.tensor(sa_strt))
        # loss_5 = criterion(None, torch.tensor(sa_end))
        # total_loss = loss_1 + loss_2 + loss_3 # + loss_4 + loss_5
        # cum_loss += total_loss.item()
        # num_steps += 1
        # total_loss.backward()
        # optimizer.step()
        # data_loader_tqdm_obj.set_description_str("Loss : %.6f" % (cum_loss / num_steps))



