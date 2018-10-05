#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:59:32 2018

@author: lockie
"""

import os
import pandas as pd
import pandas_datareader.data as web
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col

os.getcwd()
os.chdir('/Users/lockie/Desktop/Code/') 

# Read in data from csv as Yahoo! Finance depreciated ## start date 1 Jan 2011 
df1 = pd.read_csv('AAPL.csv', index_col='Date') ## Changed to GOOG.csv and TSLA.csv for other regs
df = df1.filter(items = ['Adj Close'])
df['Return'] = (df['Adj Close'] / df['Adj Close'].shift(1) -1 ) #could use np.log aswell
print(df.head())

# Import fama/french five factor data
factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
print(factors.head())
factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
factors['MKT'] = factors['MKT']/100
factors['SMB'] = factors['SMB']/100
factors['HML'] = factors['HML']/100
factors['RMW'] = factors['RMW']/100
factors['CMA'] = factors['CMA']/100

# Merge Stock and FF5 together
m = pd.merge(df,factors,left_index=True,right_index=True)
print(m.head())

# Subtract risk free rate from stock
m['Exret'] = m['Return'] - m['RF']

# CAPM regression - Was trying to find Newey-West standard errors in python
# from what I found HAC is similar. I might be wrong.
CAPM = sm.ols(formula = 'Exret ~ MKT', data=m).fit(cov_type='HAC',cov_kwds={'maxlags':1})
print(CAPM.summary())

# FF3 regression 
FF3 = sm.ols( formula = 'Exret ~ MKT + SMB + HML', data=m).fit(cov_type='HAC',cov_kwds={'maxlags':1})
print(FF3.summary())

# FF5 regression 
FF5 = sm.ols( formula = 'Exret ~ MKT + SMB + HML + RMW + CMA', data=m).fit(cov_type='HAC',cov_kwds={'maxlags':1})
print(FF5.summary())

# Storing tstats and coefficients
CAPMtstat = CAPM.tvalues
FF3tstat = FF3.tvalues
FF5tstat = FF5.tvalues

CAPMcoeff = CAPM.params
FF3coeff = FF3.params
FF5coeff = FF5.params

# DataFrame with coefficients and t-stats
results_df = pd.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                           'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

# Read DataFrame to csv file
results_df.to_csv('Sample1.csv', sep = ',')

# Created table in excel with the coefficients and tstats

# Created this below and wanted to export this table to excel, however, cannot figure
# out how to do this just yet. Displays significance stars on the coefficients which
# I wanted. 
dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                  model_names=['CAPM','FF3','FF5'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                             regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

print(dfoutput)




