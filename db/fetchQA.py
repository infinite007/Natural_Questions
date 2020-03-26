# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:54:36 2020

@author: Lokeshwar
"""

import json
from bs4 import BeautifulSoup
import pymongo


backup={}
long_answers=[]
short_answers=[]
table_list=[]
tr_list=[]
li_list=[]


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
print(myclient.list_database_names())
if('NQ' in myclient.list_database_names()):
    db=myclient['NQ']
    if('QA' in db.list_collection_names()):
        collection=db['QA']
        

def get_ans(x):
    question=x['question_text']
    backup[question]=[]
    long_answer=""
    short_answer=""
    isShortAns=False
    isLongAns=False
    
    la_start_index=x['annotations'][0]['long_answer']['start_token'] if x['annotations'][0]['long_answer']!= [] else 0
    la_end_index=x['annotations'][0]['long_answer']['end_token'] if x['annotations'][0]['long_answer']!= [] else 0
    sa_start_index=x['annotations'][0]['short_answers'][0]['start_token']if x['annotations'][0]['short_answers']!= [] else 0
    sa_end_index=x['annotations'][0]['short_answers'][0]['end_token'] if x['annotations'][0]['short_answers']!= [] else 0
    document_text=x['document_text']
    la_answer=' '.join(document_text.split()[la_start_index:la_end_index])
    soup=BeautifulSoup(la_answer,'lxml')
          
    #short answer       
    if(sa_start_index>0 and sa_end_index>0):
        isShortAns=True
        short_answer=' '.join(document_text.split()[sa_start_index:sa_end_index])
        short_answers.append(short_answer)    
      
    #long answer
    if(la_answer.split()!=[]):
        isLongAns=True
        #<p> or <dd> or <dl> 
        if(la_answer.split()[0]=='<P>' or  la_answer.split()[0]=='<Dd>' or la_answer.split()[0]=='<Dl>'):
            tag=la_answer.split()[0]
            long_answer = soup.find('p' if tag=='<P>' else 'dd' if tag=='<Dd>' else 'dl' ).text
            long_answers.append(long_answer)
            
        #<li>
        elif(la_answer.split()[0]=='<Li>'):
            li_list.append(la_answer)
            long_answer = soup.find('li').text
            long_answers.append(long_answer)     
          
        #<Ul> or <Ol>
        elif((la_answer.split()[0]=='<Ul>' or la_answer.split()[0]=='<Ol>')):
            long_answer=[item.text for item in soup.findAll('li')]
        
        elif(la_answer.split()[0]=='<Table>'):
            long_answer = soup.find('table').text
            table_list.append(la_answer)
        
        elif(la_answer.split()[0]=='<Tr>'):
            long_answer = soup.find('tr').text
            tr_list.append(la_answer)
    
    if(not(isShortAns and isLongAns)):
        rCount+=0
    

    #Create a record and dump into db     
    record={"question":question,'long_answer':long_answer,'short_answer':short_answer}
    collection.insert_one(record) #dumping into MongoDB
    backup[question].append({'long_answer':long_answer,'short_answer':short_answer})
        
   
        
with open('small_data.json','r') as f:
    for line in f:
        abc=json.loads(line)
        get_ans(json.loads(line))




## Serializing json  
#json_object = json.dumps(data, indent = 2) 
#  
## Writing to sample.json 
#with open("qa.json", "w") as outfile: 
#    outfile.write(json_object) 


#=============================================================================
# # Printing all the data inserted 
# cursor = collection.find() 
# for record in cursor: 
#     print(record)
#     break
# 
 # Printing a particular record
def fetchaRecord(question):
    aRec = collection.find({'question':question})
    for doc in aRec:
        print(doc['question'])
        print(doc['long_answer'])
        print(doc['short_answer'])
#=============================================================================
