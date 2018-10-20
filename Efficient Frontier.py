#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:37:40 2018

@author: lockie
"""

import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use Quandl to get adjusted close price
quandl.ApiConfig.api_key = 'ENTER YOUR KEY'
stocks = ['MSFT', 'AAPL', 'WMT', 'GE', 'TSLA', 'KO', 'F', 'JNJ', 'BA', 'XOM']
stockdata = quandl.get_table('WIKI/PRICES', ticker = stocks, paginate=True,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2010-1-1', 'lte': '2017-12-31' })

# Setting date as index with columns of tickers and adjusted closing price
data1 = stockdata.set_index('date')
table = data1.pivot(columns='ticker')

# Daily and annual returns of the stocks
returns_daily = table.pct_change()
returns_annual = returns_daily.mean() * 252

# Daily and annual covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 252

# Number of portolios to be generated
num_assets = len(stocks)
num_portfolios = 50000

# Empty lists to store returns, volatility and weights and sharpe ratio 
# of portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# loop of the number of portfolios to be created and respective rets, vol, 
# sharpe and weights

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# Dictionary for rets, vol and sharpe ratio of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# Adjust the original dictionary to accomodate each ticker and weight in the portfolio
for counter,stock in enumerate(stocks):
    portfolio[stock+' Weight'] = [Weight[counter] for Weight in stock_weights]

# Dictionary to DataFrame
df = pd.DataFrame(portfolio)

# Labels for columns
columns = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in stocks]

# Reorder dataframe columns
df = df[columns]

# min vol and max sharpe ratio 
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# Plot frontier, min vol, max sharpe
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='viridis', edgecolors='black', figsize=(10, 6), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='blue', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='purple', marker='D', s=200 )
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.show()