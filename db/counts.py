# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:54:36 2020

@author: Lokeshwar
"""

import json

x={}
backup={}
long_answers=[]
short_answers=[]
table_list=[]
dd_list=[]
dl_list=[]
ol_list=[]
ul_list=[]
tr_list=[]
        

def get_ans(x):
    question=x['question_text']
    backup[question]=[]
    
    
    la_start_index=x['annotations'][0]['long_answer']['start_token'] if x['annotations'][0]['long_answer']!= [] else 0
    la_end_index=x['annotations'][0]['long_answer']['end_token'] if x['annotations'][0]['long_answer']!= [] else 0
    #sa_start_index=x['annotations'][0]['short_answers'][0]['start_token']if x['annotations'][0]['short_answers']!= [] else 0
    #sa_end_index=x['annotations'][0]['short_answers'][0]['end_token'] if x['annotations'][0]['short_answers']!= [] else 0
    document_text=x['document_text']
    la_answer=' '.join(document_text.split()[la_start_index:la_end_index])
    #long answer
    if(la_answer.split()!=[] and la_answer.split()[0]=='<Dd>'):
        dd_list.append(la_answer)
    elif(la_answer.split()!=[] and la_answer.split()[0]=='<Dl>'):
        dl_list.append(la_answer)
    elif(la_answer.split()!=[] and la_answer.split()[0]=='<Ol>'):
        ol_list.append(la_answer)
    elif(la_answer.split()!=[] and la_answer.split()[0]=='<Ul>'):
        ul_list.append(la_answer)
    elif(la_answer.split()!=[] and la_answer.split()[0]=='<Tr>'):
        tr_list.append(la_answer)
        
with open('small_data.json','r') as f:
    for line in f:
        abc=json.loads(line)
        get_ans(json.loads(line))

# {'<Dd>', '<Dl>', '<Li>', '<Ol>', '<P>', '<Table>', '<Tr>', '<Ul>'}


#
#li_list.count('<P>') *
#Out[3]: 32333
#
#li_list.count('<Dd>') *
#Out[4]: 32
#
#li_list.count('<Dl>') *
#Out[5]: 105
#
#li_list.count('<Li>') *
#Out[6]: 1472
#
#li_list.count('<Ol>') *
#Out[7]: 101
#
#li_list.count('<Table>')
#Out[8]: 8427
#
#li_list.count('<Tr>')
#Out[9]: 664
#
#li_list.count('<Ul>') *
#Out[10]: 1240

#45097 - Empty , short_ans=[] , long_ans start and end index -1
        
        
        
        
from bs4 import BeautifulSoup

soup=BeautifulSoup(tr_list[1],'lxml')
alpha=soup.find('tr').text