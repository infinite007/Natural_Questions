# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:12:06 2020

@author: rajender
"""

import json
import jsonlines

class sample_extraction():
    def __init__(self,path):
        #self.file=open("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/simplified-nq-train.jsonl")
        self.file=open(path)
    
    def extracting_sample(self,destination_path):
        list_sample=[]
        i=0
        for line in self.file:
            if (i> 50000):
                if(i!=100000):
                 l=json.loads(line)
                 list_sample.append(l)
                 i=i+1
                if(i==100000):
                 break
            i=i+1
             
        json_object=json.dumps(list_sample)
        
        #with open("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/test_sample.jsonl2","w") as outfile:
        

        with jsonlines.open(destination_path, 'w') as writer:
            writer.write_all(json_object)
#        with open(destination_path,"w") as outfile:
#            for entry in json_object:
#                json.dump(entry, outfile)
#                outfile.write('\n')
        return json_object
        


