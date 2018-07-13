'''
Created on Jan 27, 2018

@author: Arya Roy
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from pandas.core import window
from Crypto.Util.number import size
from numpy import cumprod
from datetime import datetime
import os
from fileinput import filename

# class FileMaker(object):
#     def __init__(self):
#         self.stocks_data = pd.DataFrame(columns=['Date'])
#         self.stocks_data = self.stocks_data.set_index('Date')
# def init_load(self):
#file location to which the  havcopy is dowloaded
base= 'C:/Getbhavcopy/data/NSE-EOD'
filenames = os.listdir(base)
dates = [f.replace('-NSE-EQ.txt','')for f in filenames]
r = len(dates)
df0 = pd.read_csv('E:/Bhavcopy/data/NSE-EOD/2018-01-16-NSE-EQ.txt')
w = len(df0['<ticker>'])
data = np.zeros((r,w), dtype=np.float64)
def my_func(i,symbol):
    '''
    function to return the closing price series for the stock 
    having the ticker - symbol- which is an input given to this function
    '''
    df = pd.read_csv('/'.join((base,filenames[i])))
    if not df.loc[df['<ticker>']==symbol]['<close>'].empty:
        return df.loc[df['<ticker>']==symbol]['<close>']
    else :
        return None
# loop to create the data table having the closing prices of all the tickers for each trading day    
for i in range(r):
    for j in range(w):
        data[i,j]=my_func(i,df0['<ticker>'][j])
        print(data[i,j],i,j)

df = pd.DataFrame(data,columns=df0['<ticker>'],index=dates)
df1 = pd.read_csv('E:/Bhavcopy/data/final.csv',index_col=0)
df = pd.concat([df1,df])
df.to_csv('E:/Bhavcopy/data/final.csv',header=True)   
                
#             for (dirpath, dirnames, filenames) in os.walk(base):
#                 
#                 for f in filenames:
#                     print(f)
#                     e = os.path.join(str(dirpath),str(f))
#                     df = pd.read_csv(e)
# #                     l = ['DCBBANK','PNCINFRA','REPCOHOME','GSPL','CHENNPETRO','JCHAC','JYOTHYLAB','CGPOWER','SCHAND','SATIN']
#                     for symbol in df['<ticker>']:
#                         mask = df.loc[df['<ticker>']==symbol]['<close>']
#                         print(mask)
#                         if str(mask)!='-'and not mask.empty:
#                             with open('E:/Bhavcopy/data/final.csv','w') as f :
#                                 self.stocks_data.loc[df['<date>'][0],symbol] = float(mask)                            
#                                 self.stocks_data.to_csv(f,header=True)
# #     def add_rows(self,date): 
# #         date = pd.to_datetime(date)  
# #         date = datetime.strftime(date,'%Y-%m-%d')
# f = FileMaker()
# f.init_load()
# my_func()         
          