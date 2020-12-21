# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:13:08 2020

@author: sudhe
"""
"""
Time Series forcasting for one minite time frame in daily data
"""

import pandas as pd
data=pd.read_csv(r'C:\Users\sudhe\Desktop\Stock Market Research\Data\Intraday 1 Min Data\2019\2019 SEP NIFTY.txt',sep=',',header=None)

data.columns
data.drop(columns=[0,7,8],axis=1,inplace=True)
data.dtypes

data[data[1]==20190903]

working_data=data[data[1]==20190903]

working_data.columns=['date','time','open','high','low','close']

working_data=working_data[1:-4]

working_data.reset_index()

working_data['date']=working_data['date'].apply(lambda x:str(x))
working_data['date']=working_data['date'].apply(lambda x:(x[:4]+'-'+x[4:6]+'-'+x[6:]))


working_data['date'].iloc[0][:4]+'-'+working_data['date'].iloc[0][4:6]+'-'+working_data['date'].iloc[0][6:]

def date_time_joiner(x,y):
    return x+" "+y

working_data['date_time']=working_data[['date','time']].apply(lambda row:date_time_joiner(row['date'],row['time']),axis=1)


    
working_data['date_time'].iloc[0]
from datetime import datetime

#datetime.strptime(working_data['date_time'].iloc[0],'%Y-%m-%d %H:%M')
working_data['date_time']=working_data['date_time'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M'))

working_data.set_index('date_time',drop=True,inplace=True)
working_data.drop(columns=['date','time'],axis=1,inplace=True)

working_data['close'].plot()

working_data['open_close']=working_data['open']-working_data['close']


from fitter import Fitter

f = Fitter(working_data['open_close'])
f.fit()
f.summary()
