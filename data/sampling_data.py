# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:25:36 2020

@author: krish
"""

import pandas as pd
import json

i=50000
list_sample=[]
with open("C:/ASU/CSE_551_FOA/v1.0-simplified_simplified-nq-train/simplified-nq-train.jsonl") as file:
    for line in file:
        if (i!=100000):
            l=json.loads(line)
            list_sample.append(l)
            i=i+1
        if (i==50000):
            break

sample={'data':[]}
test_sample_df=pd.DataFrame(sample)
json_object=json.dumps(list_sample)

with open("C:/ASU/CSE_551_FOA/v1.0-simplified_simplified-nq-train/test_sample.jsonl", "w") as outfile: 
    outfile.write(json_object)
            