# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:26:03 2020

@author: Lokeshwar
"""

#Table
from bs4 import BeautifulSoup
longanswer=""
soup=BeautifulSoup(longanswer,'lxml')
if(longanswer.split()[0]=='<Table>'):
    table=soup.find('table')
    rows=table.find_all('tr')
    for tr in rows:
        td=tr.find_all('td')
        row = ' '.join([i.text.strip() for i in td])
        print(row)
    


html_string = '''
      <table>
            <tr>
                <td> Hello! </td>
                <td> Table </td>
            </tr>
            <tr>
                <td> Hello123! </td>
                <td> Table123 </td>
            </tr>
              <tr>
                <td> Hello123! </td>
                <td> Table123 </td>
            </tr>
        </table>
    '''
soup=BeautifulSoup(html_string,'lxml')
table=soup.find('table')
rows=table.find_all('tr')
long_answer_list=[]
for tr in rows:
    td=tr.find_all('td')
    row = ' '.join([i.text.strip() for i in td])
    long_answer_list.append(row)