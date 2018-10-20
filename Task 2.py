#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:13:38 2018

@author: lockie
"""

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# Use Quandl to get adjusted close price
quandl.ApiConfig.api_key = 'ENTER YOUR KEY' # Have to make a Quandl account 
stocks = ['MSFT', 'AAPL', 'WMT', 'GE', 'KO', 'F', 'JNJ', 'BA', 'XOM', 'IBM']
stockdata = quandl.get_table('WIKI/PRICES', ticker = stocks, paginate=True,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '1995-1-1', 'lte': '2010-12-31' })

# Setting date as index with columns of tickers and adjusted closing price

df = stockdata.pivot(index = 'date',columns='ticker')
df.index = pd.to_datetime(df.index)
df = df.resample('1M').mean()
df = df.pct_change() 
df = df.rename(columns = {'adj_close':'rets'})
df.head()

nos = len(stocks)
window = 60 # 60 month historical window
period = len(df)
dfrolling = df[window:]
x = period - window 

Optw = [] # empty list
Minvarw = [] # empty list

for i in range(window,period):
    
    rets=df[1:i] # remove the first NaN observation
    rets.mean() * 12
    rets.cov() * 12
    
    weights = np.random.random(nos)
    weights /= np.sum(weights) 
    weights
    np.sum(rets.mean() * weights ) * 12
      
    # expected portfolio return
    
    np.dot(weights.T, np.dot(rets.cov() * 12, weights))
    np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights))) 

    """Efficient Frontier""" 
        ''' 
        Port return
        Port vol
        Assume Sharpe ratio = 0 
        '''
    
    def statistics(weights):
        
        weights = np.array(weights)
        pret = np.sum(rets.mean() * weights) * 12
        pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 12, weights)))
        return np.array([pret, pvol, pret / pvol])
    
    """ Optimal Sharpe Ratio Weights"""
    
    def negSharpeRatio(weights):
        return -statistics(weights)[2]
    
    # constraint that weights must add up to 1 
    
    constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # weight values must be within 0 and 1
    
    bounds = tuple((0, 1) for x in range(nos))
    
    def min_portfolio_vola(weights):
        return statistics(weights)[1]
    
    """ Find Max Sharpe Ratio Portfolio """
    
    opts = sco.minimize(negSharpeRatio, nos * [1. / nos,], method='SLSQP',
                        bounds = bounds, constraints = constraint)
    sharpew = opts['x'].round(3)
    sharpewstats = statistics(opts['x']).round(3)
    Optw.append(sharpew) # append weights of opt Sharpe to list
        
    """ Find Min Variance Portfolio Weights"""
    
    def min_func_variance(weights):
        return statistics(weights)[1] ** 2
    
    """ Find Min Variance Portfolio"""
    
    optv = sco.minimize(min_func_variance, nos * [1. / nos,], method='SLSQP',
                        bounds=bounds, constraints=constraint)
    Min_Varw = optv['x'].round(3)
    
    # statistics: Expected return, vol and Sharpe
    
    minvarwstats = statistics(optv['x']).round(3)
    Minvarw.append(Min_Varw) # append weights of min var to list
       
# Optimal Sharpe Returns
    
OptimalSharperolling = (dfrolling[0:x].values)*Optw # opt sharpe return indvidual stocks
op = pd.DataFrame(OptimalSharperolling, index=dfrolling[0:x].index) # array to DataFrame
op.columns = [list(dfrolling.columns.values)] # assign column names to weights
Sharpewrets = op.sum(1) # opt Sharpe return portfolio

# Minimum Variance Returns

MVrolling = (dfrolling[0:x].values)* Minvarw # min var return indvidual stocks
mv = pd.DataFrame(MVrolling, index=dfrolling[0:x].index) # array to DataFrame
mv.columns = [list(dfrolling.columns.values)] # assign column names to weights
MVwrets = mv.sum(1) # min variance return portfolio

