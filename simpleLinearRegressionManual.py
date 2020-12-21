import pandas as pd
import numpy as np

def simpleRegressionCoeefficients(X:np.array,Y:np.array)->list:
    data=pd.DataFrame()
    data['X']=pd.Series(X)
    data['Y']=pd.Series(Y)
    data['Xi-Xbar_wholesquare']=np.square(data['X']-np.mean(data['X']))
    data['X-Xbar*Y-Ybar']=(data['X']-np.mean(data['X']))*(data['Y']-np.mean(data['Y']))
    b1=np.sum(data['X-Xbar*Y-Ybar'])/np.sum(data['Xi-Xbar_wholesquare'])
    b0=np.mean(data['Y'])-np.mean(data['X'])*b1

    return list([b0,b1])

def rmse(X:np.array,Y:np.array)->float:
    data=pd.DataFrame()
    data['X']=pd.Series(X)
    data['Y']=pd.Series(Y)
    data['Xi-Xbar_wholesquare']=np.square(data['X']-np.mean(data['X']))
    data['X-Xbar*Y-Ybar']=(data['X']-np.mean(data['X']))*(data['Y']-np.mean(data['Y']))
    b1=np.sum(data['X-Xbar*Y-Ybar'])/np.sum(data['Xi-Xbar_wholesquare'])
    b0=np.mean(data['Y'])-np.mean(data['X'])*b1
    data['Ypred']=b0+b1*data['X']
    data['error']=data['Y']-data['Ypred']

    return np.sqrt(np.sum(np.square(data['error']))/(len(data)-1))
