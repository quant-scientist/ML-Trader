'''
Created on Jan 22, 2018

@author: Arya Roy
'''
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from pandas.core import window
from Crypto.Util.number import size
from numpy import cumprod
from astropy.constants.codata2010 import alpha
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
from dask.array.ufunc import sqrt

df1 = pd.read_csv('E:/Bhavcopy/data/final.csv') #file created using filemaker.py after downloading data using Bhavcopy
# I = df1['Date']
# print(df1.columns)
df1['Date']= pd.to_datetime(df1['Date'])
df1['Date'] = df1['Date'].dt.strftime('%Y/%m/%d')
df1 = df1.set_index('Date')
df1 = df1.filter(items=['HDFCBANK','ICICIBANK','LT','HDFC','INFY','SBIN','MARUTI',  #large cap stocks filtered out from available stocks
                        'TATASTEEL','GRASIM','KOTAKBANK','INDUSINDBK','ITC','RELIANCE',
                        'VEDL','GAIL','HAVELLS','ADANIPORTS','TATAGLOBAL','BAJAJ-AUTO',
                        'BHARTIARTL','TATAMOTORS','NAUKRI','HINDUNILVR','CEATLTD','RAYMOND',
                        'HINDPETRO','VOLTAS','EXIDEIND','LUPIN','SUNPHARMA','FEDERALBNK','YESBANK',
                        'TCS','TATACHEM','AXISBANK','AUROPHARMA','TORNTPHARM','APOLLOHOSP','PETRONET',
                        'VINATIORGA','EMAMILTD','POWERGRID','HCLTECH','DABUR','JKCEMENT','ICICIGI','PFC',
                        'COCHINSHIP','CESC','ZEEL','CONCOR','IOC','ASIANPAINT'])
# df1=df1.filter(items=['TATASTEEL'])
df2 = pd.read_csv('C:/Users/personalPC/Desktop/my_project/QTrader/tbills.csv',sep='\t')
# print(df2.dtypes)
df2['Date']= pd.to_datetime(df2['Date'])
df2['Date'] = df2['Date'].dt.strftime('%Y/%m/%d')
df2 = df2.set_index('Date')
df2.columns = ['tPrice', 'tOpen', 'tHigh', 'tLow', 'tChange %']
df2['tPrice'] = pd.to_numeric(df2['tPrice'],errors='coerce')
# df2.index = pd.DatetimeIndex(df2.index)
# df2['tPrice'] = df2['tPrice'].interpolate()

# df2['Date']= pd.to_datetime(df2['Date'])


  
class QTrader(object):
    def __init__(self):
        self.stock_data = pd.concat([df1,df2],axis=1,join_axes=[df1.index]).sort_index()
        self.stock_data['tPrice'] = self.stock_data['tPrice'].interpolate()
#         print(self.stock_data.loc['2014/01/14'])
#         self.returns = pd.concat([df1.sort_index().sample(n=5,axis=1).rolling(window=2).apply(lambda x: x[1]/x[0]-1),
#             (self.stock_data['tPrice']/100+1)**(1/252)-1],axis=1)
#         print(self.returns.head(4))
        self.returns = pd.concat([df1.sort_index().loc[:,['VEDL','AUROPHARMA','TATAGLOBAL']].rolling(window=2).apply(lambda x: x[1]/x[0]-1),
            (self.stock_data['tPrice']/100+1)**(1/252)-1],axis=1)
#         self.returns = pd.concat([df1.sort_index().rolling(window=2).apply(lambda x: x[1]/x[0]-1),
#         (self.stock_data['tPrice']/100+1)**(1/252)-1],axis=1)
#         print(self.returns.head(4))
        x = len(self.returns.columns)
#         data preprocessing steps given below
        for i in range(x-1):
            self.returns['_'.join(['risk_adjusted',str(i)])]=self.returns[self.returns.columns[i]]-self.returns['tPrice']
#             print(self.returns['_'.join(['risk_adjusted',str(i)])].loc['2014/01/07'])
        for i in range(x-1):
            self.returns['_'.join(['risk_adjusted_moving',str(i)])] = self.returns['_'.join(['risk_adjusted',str(i)])].rolling(window=12).apply(lambda x: x.mean())
        for i in range(x-1):    
            self.returns['_'.join(['risk_adjusted_stdev',str(i)])] = self.returns['_'.join(['risk_adjusted',str(i)])].rolling(window=12).apply(lambda x: x.std())
        for i in range(x-1):    
            self.returns['_'.join(['risk_adjusted_high',str(i)])] = self.returns['_'.join(['risk_adjusted_moving',str(i)])] + 0.1 * self.returns['_'.join(['risk_adjusted_stdev',str(i)])]
        for i in range(x-1):    
            self.returns['_'.join(['risk_adjusted_low',str(i)])] = self.returns['_'.join(['risk_adjusted_moving',str(i)])] - 0.1 * self.returns['_'.join(['risk_adjusted_stdev',str(i)])]
        for i in range(x-1):    
            self.returns['_'.join(['state',str(i)])] = (self.returns['_'.join(['risk_adjusted',str(i)])] > self.returns['_'.join(['risk_adjusted_low',str(i)])]).astype('int') - \
                                (self.returns['_'.join(['risk_adjusted',str(i)])]< self.returns['_'.join(['risk_adjusted_high',str(i)])]).astype('int')
                                 
# pd.qcut(self.returns.sharpe_moving, 10, labels=range(10))
    def buy_and_hold(self,dates):
        '''
        Function to create the buy and hold portfolio
        '''
        l=len(self.returns.columns)
        i=int((l-1)/7)
        return pd.DataFrame(1, index=dates,columns=self.returns.columns[:i])
    def buy_tbills(self,dates):
        '''
        Function to create the buy and hold sovereign bond portfolio
        '''
        l=len(self.returns.columns)
        i=int((l-1)/7)
        return pd.DataFrame(0,index=dates,columns=self.returns.columns[:i])
    def random(self,dates):
        '''
        Function to create the random strategy portfolio
        '''
        r = len(dates)
        l = len(self.returns.columns)
        i = int((l-1)/7)
        return pd.DataFrame(np.random.randint(-1,2,size=[r,i]),index=dates,columns=self.returns.columns[:i])
    def evaluate(self,holdings):
        '''
        Function to find the cumulative excess return of the portfolio
        '''
        l=len(self.returns.columns)
        i = int((l-1)/7)
#         return pd.DataFrame(self.returns[self.returns.columns[i]].ix[holdings.index]+holdings.values*self.returns.ix[holdings.index][self.returns.columns[i+1:2*i+1]].values+1,columns=self.returns.columns[:i],
#                            index=holdings.index).apply(lambda x: x.cumprod()).sum(axis=1)
        print(holdings.shape)
        print(self.returns.loc[holdings.index].shape)
        return pd.DataFrame(holdings.values*self.returns.loc[holdings.index][self.returns.columns[i+1:2*i+1]].values+1,columns=self.returns.columns[:i],
                            index=holdings.index).apply(lambda x: x.cumprod()).sum(axis=1)                 
#         return pd.DataFrame(self.returns[self.returns.columns[i]].to_frame().loc[holdings.index].values+holdings.values*self.returns.loc[holdings.index,self.returns.columns[i+1:2*i+1]].values+1,index=holdings.index).apply(lambda x: x.cumprod()).sum(axis=1)
    def evaluate_ind(self,holdings,t):
        '''
        Function to find the cumulative excess return of individual stock
        '''
        l=len(self.returns.columns)
        i = int((l-1)/7)
        return (holdings.values*self.returns.loc[holdings.index,self.returns.columns[i+t+1]]+1).to_frame().apply(lambda x: x.cumprod())
#     def kelly_pfolio(self,holdings):
#         l=len(self.returns.columns)
#         i = int((l-1)/7)
#         date = holdings.index[-1] 
        
    def q_holdings(self,training_indexes, testing_indexes,alpha=.5):
        '''
        Function to implement the RL algorithm in portfolio building
        Input : training index list, testing index list, alpha i.e learning rate
        '''
        l=len(self.returns.columns)
        l1 = int((l-1)/7)
        factors = pd.DataFrame({'action': 0, 'reward': 0, 'state': 0}, index=training_indexes)
        portf_factors = pd.DataFrame(np.zeros((len(training_indexes),l1)),index=training_indexes,columns=[self.returns.columns[:l1]])
        portf_testing = pd.DataFrame(np.zeros((len(testing_indexes),l1)),index=testing_indexes,columns=[self.returns.columns[:l1]])
        for m in range(l1):
            # Initialize Q matrix
            q = {0: {1:0, 0:0, -1:0}}
            
            # For Dyna-Q
            T = np.zeros((3, 3, 3)) + 0.00001
            R = np.zeros((3,3))
    
            # Episodes
            for i in range(100):
                last_row, last_date = None, None
                count =0    
                for date, row in factors.iloc[::-1].iterrows():
                    return_data = self.returns.loc[date]
                    count += 1
                    # discount factor 
                    discount =  math.sin(math.exp(-math.radians(count)))
#                     if return_data['_'.join(['state',str(m)])] not in q :                        
#                         q[return_data['_'.join(['state',str(m)])]] = {1: 0, 0:0, -1:0} 
                    if last_row is None or np.isnan(return_data['_'.join(['state',str(m)])]):
                        state = 0
                        reward = 0
                        action = 0
                    else:
#                         if row.state not in q :
#                             q[row.state]=   {1: 0, 0:0, -1:0}
                        state = int(return_data['_'.join(['state',str(m)])])
                        if random.random() > 0.001:
                            try :
                                action = max(q[state], key=q[state].get)
                            except:
                                q[state]= {1: 0, 0:0, -1:0}
                                action = max(q[state], key=q[state].get)
                                    
                        else:
                            action = random.randint(-1,1)
    
                        reward = last_row.action * (return_data[self.returns.columns[m]] - return_data['tPrice'])
    
                        
    
                        factors.loc[date, 'reward'] = reward
                        factors.loc[date, 'action'] = action
                        factors.loc[date, 'state'] = return_data['_'.join(['state',str(m)])]
                        try:
                            print(max(q[row.state]))
                        except KeyError :
                            q[row.state]= {1: 0, 0:0, -1:0}
                        # update the reward function   
                        update = alpha * (factors.loc[date, 'reward'] + discount * max(q[row.state].values()) - q[state][action])
                        if not np.isnan(update):
                            q[state][action] += update
    
                        # Dyna
                        action_idx = int(last_row.action+1)
                        state_idx = int(last_row.state+1)
                        new_state_idx = int(state+1)
    
                        T[state_idx][action_idx][new_state_idx] += 1
                        R[state_idx][action_idx] = (1 - alpha) * R[state_idx][action_idx] + alpha * reward
    
                    last_date, last_row = date, factors.loc[date]
    
                for j in range(100):
                    state_idx = random.randint(0,2)
                    action_idx = random.randint(0,2)
                    new_state = np.random.choice([-1, 0, 1], 1, p=T[state_idx][action_idx]/T[state_idx][action_idx].sum())[0]
                    r = R[state_idx][action_idx]
                    try :
                        q[state][action] += alpha * (r + discount * max(q[new_state].values()) - q[state][action])
                    except KeyError:
                        q[new_state] =  {1: 0, 0:0, -1:0}
                        q[state][action] += alpha * (r + discount * max(q[new_state].values()) - q[state][action])
            portf_factors[self.returns.columns[m]]= factors.action
            testing = pd.DataFrame({'action': 0, 'state': 0}, index=testing_indexes)
            testing['state'] = self.returns['_'.join(['state',str(m)])]
            testing['action'] = testing['state'].apply(lambda state: max(q[state], key=q[state].get))
            portf_testing[self.returns.columns[m]]=testing.action
        # Sharpe ratio of the RL trading model    
        sharpe = self.sharpe(portf_factors)
        print("We get an internal sharpe ratio of {}".format(self.sharpe(portf_factors)))           
        print(self.sharpe(portf_testing))
        return portf_testing
    def sharpe(self, holdings):
        '''
        Function to calculate the portfolio Sharpe ratio
        '''
        l=len(self.returns.columns)
        l1 = int((l-1)/7)
#         print(holdings.shape, (self.returns[self.returns.columns[:l1]].ix[holdings.index].values - self.returns['tPrice'].to_frame().ix[holdings.index]).shape)
        returns = pd.DataFrame(holdings.values * (self.returns[self.returns.columns[:l1]].ix[holdings.index].apply(lambda x : x-((abs(x))/200)-self.returns['tPrice'].ix[holdings.index])),columns=self.returns.columns[:l1],
                           index=holdings.index)
        returns['mean']=returns.apply(lambda x: np.nanmean(x),axis=1)
        return np.nanmean(returns['mean'])/np.nanstd(returns['mean'])
    def drawdown(self,holdings):
        '''
        Function to calculate the maximum drawdown and maximum drawdown duration
        '''
        l=len(self.returns.columns)
        l1 = int((l-1)/7)
        maxdd_a = []
        maxddd_a = []
        for i in range(l1) :
            dd = np.zeros((len(holdings.index),1))
            ddd = np.zeros((len(holdings.index),1))
            highwatermark = np.zeros((len(holdings.index),1))
            cumret = self.evaluate_ind(holdings.iloc[:,i],i)
            print(cumret.tail(2))
            for j in range(1,len(holdings.index)):
#                 print(cumret.loc[holdings.index[j],cumret.columns[0]])
                highwatermark[j]=max(highwatermark[j-1],cumret.loc[holdings.index[j],cumret.columns[0]])
                dd[j]=(1+highwatermark[j])/(1+cumret.loc[holdings.index[j]])-1
                if dd[j]==0:
                    ddd[j]=0
                else :
                    ddd[j]=ddd[j-1]+1
            maxdd = np.amax(dd,axis=0)
            maxddd = np.amax(ddd,axis=0)
            maxdd_a.append(maxdd)
            maxddd_a.append(maxddd)                
        return maxdd_a, maxddd_a 
    def get_CAGR(self,holdings):
        '''
        Functon to calculate the compound annual growth rate (CAGR)
        '''
        l=len(self.returns.columns)
        l1 = int((l-1)/7)
        di = holdings.index[0]
        df = holdings.index[-1]
        y = int(df[:4])-int(di[:4])
        di1 = (dt.strptime(df,'%Y/%m/%d')+relativedelta(years=-y))
        di1_f = min(holdings.index,key=lambda x: abs(dt.strptime(x,'%Y/%m/%d')-di1))
        e = self.evaluate(holdings)
#         print(e.loc[di1_f])
        try:
            return (e.loc[df]/e.loc[di1_f])**(1/y)-1
        except:
            return 0
    def support_pts(self,holdings):
        '''
        Function to calculate the inflection price points of individual stocks based on volatility indicator
        '''
        l=len(self.returns.columns)
        l1 = int((l-1)/7)
        cols=pd.MultiIndex.from_product([self.returns.columns[:l1],['Daily Return','Price']])
        support_table = pd.DataFrame(np.ones((10,len(cols))),columns=cols,index=range(10))
        for i in range(l1):
            t= self.returns.columns[i]
            tmp = self.returns.sort_values(by=[t],na_position='first').filter(items=[t])
            tmp.columns=['Daily Return']
            print(tmp.tail(4))
            tmp1 = self.stock_data.filter(items=[t])
            tmp1.columns=['Price']
            print(tmp1.tail(4))
            tmp = pd.concat([tmp,tmp1],axis=1,join_axes=[tmp.index])
            tmp.columns = ['Daily Return','Price']
            support_table[t]=tmp
        return support_table.tail(10)    
            
    def graph_portfolios(self):
        '''
        Function to create the visualization of the RL trader daily returns as well as 
        calculate the output parameters of the RL trader
        '''
        l=len(self.returns.columns)
        i=int((l-1)/7)
        testpoint = int(len(self.returns.index)*6/7)
        trainpoint = int(len(self.returns.index)*4/7)
        # training index list
        training_indexes = self.returns.index[:testpoint] 
        #testing index list
        testing_indexes = self.returns.index[testpoint:testpoint+283]
#             midpoint = int(len(self.returns.index)/2)
#             training_indexes = self.returns.index[:midpoint]
#             testing_indexes = self.returns.index[midpoint:]
        types = ['buy_and_hold','buy_tbills','random','q_holdings'] 
        cols = pd.MultiIndex.from_product([types,self.returns.columns[:i]])
        portfolios = pd.DataFrame(np.ones((len(testing_indexes),len(cols))),columns=cols,index=testing_indexes)
        portfolio_values = pd.DataFrame(np.ones((len(testing_indexes),len(types))),columns=types,index=testing_indexes)
        for s in types :
            if s!='random':
                if s!= 'q_holdings':
                    portfolios[s]= getattr(self,s)(testing_indexes)
                else :
                    portfolios[s]=getattr(self, s)(training_indexes,testing_indexes)
                portfolio_values[s]=self.evaluate(portfolios[s])
            else :
                y= pd.Series(np.zeros(len(testing_indexes)),index=testing_indexes)
                for i in range(20):
                    portfolios[s]= getattr(self,s)(testing_indexes)
                    x =  self.evaluate(portfolios[s])  
                    y= [m+n for m,n in zip(x,y)]
                portfolio_values[s]=[m/20 for m in y]    
#                     print(portfolio_values[s])
        # create dataframes to be saved
#             df1 = portfolios.random
#             df2 = portfolio_values.random
#             print(df2.head(2))
#             s = (self.sharpe(portfolios.random))*((252)**0.5)
#             mdd_a, mddd_a = self.drawdown(portfolios.random)
#             cagr = self.get_CAGR(portfolios.random)
#             supp = self.support_pts(portfolios.random)
#             print(supp.head(2))
        df1 = portfolios.q_holdings
        df2 = portfolio_values.q_holdings
        df3 = portfolio_values.random
        # Annual Sharpe ratio of the RL trader
        s = (self.sharpe(portfolios.q_holdings))*((252)**0.5)
        # Maximum drawdown, maximum drawdown duration of the RL trader
        mdd_a, mddd_a = self.drawdown(portfolios.q_holdings)
        # CAGR of the RL trader
        cagr = self.get_CAGR(portfolios.q_holdings)
        #inflection points of the individual stocks in the portfolio
        supp = self.support_pts(portfolios.q_holdings)
        df4 = pd.DataFrame({'Sharpe': s,'Max Drawdown':mdd_a, 'Max Drawdown Duration': mddd_a,'CAGR':cagr})
#             df4 = pd.DataFrame(self.returns.loc[portfolios.random.index][self.returns.columns[i]].values,columns=[self.returns.columns[i]],
#                            index=portfolios.random.index)
#             df5 = pd.DataFrame(portfolio_values.index,columns=['Return Dates'],index=range(len(portfolio_values.index)))
#             df6 = pd.DataFrame(portfolios.random.index,columns=['Holdings Dates'],index=range(len(portfolios.random.index)))
        #Excel writer
        writer =pd.ExcelWriter(''.join(['E:/Bhavcopy/data/',testing_indexes[-1].replace('/','-'),'.xlsx']),engine='xlsxwriter')
        frames = {'folio':df1,'return_tot':df2, 'random':df3,'specs':df4,'Support':supp}
        for sheet, frame in frames.items():
            frame.to_excel(writer,sheet_name=sheet)
        writer.save()                                    
        portfolio_values[['buy_tbills','random','q_holdings']].plot()
#             portfolio_values[['q_holdings']].plot()
        #visualiation of the daily returns of the RL trader and the buy abd hold portfolio
        plt.annotate("Buy and hold sharpe: {}\n QTrader sharpe: {}".format(self.sharpe(portfolios.buy_and_hold), self.sharpe(portfolios.q_holdings)), xy=(0.25, 0.95), xycoords='axes fraction')
        plt.show()       
        print(self.get_CAGR(portfolios.q_holdings))       
                                
Q = QTrader()
Q.graph_portfolios()

