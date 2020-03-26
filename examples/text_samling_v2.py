# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:12:06 2020

@author: gpunjala
"""

import json

''' path: this variable is provide with source path
    start_idx: numerical value of start index
    end_idx: numerical value of end index
    destination_path: This variable is provided with local path where output is stored
'''
class sample_extraction():
    def __init__(self,path,start_idx,end_idx):
        #self.file=open("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/simplified-nq-train.jsonl")
        self.file=open(path)
        self.start_idx=start_idx
        self.end_idx=end_idx
    
    def extracting_sample(self,destination_path):
        list_sample=[]
        i=1
        with open(destination_path,"w") as outfile:
            for line in self.file:
                if (i> self.start_idx):
                    if(i!=self.end_idx):
                     l=json.loads(line)
                     list_sample.append(l)
                     outfile.write(line)
#                     outfile.write('\n')
                     i=i+1
                    if(i==(self.end_idx+1)):
                     break
             
        
             
        json_object=json.dumps(list_sample)
        
        #with open("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/test_sample.jsonl2","w") as outfile:
        
        
        return list_sample,json_object,l

input_data_obj=sample_extraction("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/simplified-nq-train.jsonl",0,50000)
list_sample,json_object,l=input_data_obj.extracting_sample("C:/Users/rajender/Downloads/v1.0-simplified-simplified-nq-train/sample_file1.jsonl")

#for entry in json_object:
#    print(entry)

        


