# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:25:36 2020

@author: krish
"""

import json

class sample_extraction:
    def __init__(self,path):
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
        with open(destination_path,"w") as outfile:
            outfile.write(json_object)
        return json_object
            
