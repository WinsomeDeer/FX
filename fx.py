import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import product
from typing import Union
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

''' 
Data sources:

From 5/12/22 - 14/05/23.

GBP/USD data :- https://www.marketwatch.com/investing/currency/gbpusd
EUR/USD data :- https://www.marketwatch.com/investing/currency/eurusd

'''

# Load the data.
GBP_df = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\GBP.csv')
GBP_df.rename(columns = {'Open':'GBP/USD - Open'}, inplace = True)
GBP_df.index = pd.to_datetime(GBP_df['Date'])
GBP_df = GBP_df.drop('Date', axis = 1)

EUR_df = pd.read_csv(r'C:\Users\patri\Desktop\Python\Data\EUR.csv')
EUR_df.rename(columns = {'Open':'EUR/USD - Open'}, inplace = True)
EUR_df.index = pd.to_datetime(EUR_df['Date'])
EUR_df = EUR_df.drop('Date', axis = 1)

print(EUR_df.shape)

# Check NaNs.
print(GBP_df.head())
print(EUR_df.head())

# Plot the data.
fig, ax = plt.subplots()
ax.set_xlabel('Date')
ax.set_ylabel('GBP/USD')
ax.plot(GBP_df['GBP/USD - Open'])
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel('Date')
ax.set_ylabel('EUR/USD')
ax.plot(EUR_df['EUR/USD - Open'])
plt.show()

# Check if the data needs differencing.
def adfuller_test(data):
    res = adfuller(data)
    labels = ['ADF Statistic', 'p-value', '#lags used', 'Number of Observations Used']
    for value, label in zip(res, labels):
        print(label + ': ' + str(value))
    if res[1] <= 0.05:
        print('Strong evidence against the null hypothesis, reject the null hypothesis. Data is stationary.')
    else:
        print('Weak evidence against null hypothesis, time series has a unit root. Data is non-stationary.')

adfuller_test(GBP_df['GBP/USD - Open'])
adfuller_test(EUR_df['EUR/USD - Open'])

# Data is non stationary - as expected.
GBP_diff = np.diff(GBP_df['GBP/USD - Open'], n = 1)
EUR_diff = np.diff(EUR_df['EUR/USD - Open'], n = 1)

adfuller_test(GBP_diff)
adfuller_test(EUR_diff)

# Combine the data for convenience.
df_concat = pd.concat([GBP_df['GBP/USD - Open'], EUR_df['EUR/USD - Open']], axis = 1, join = 'inner')
print(df_concat.head())

# Data is now stationary - now find the best model.
def optimize_VAR(endog: Union[pd.Series, list]):
    
    res = []

    for i in range(10):
        try:
            model = VARMAX(endog, order = (i, 0), simple_differencing = False).fit(disp = False)
        except:
            continue
        aic = model.aic
        res.append([i, aic])
    res_df = pd.DataFrame(res)
    res_df.columns = [i, 'AIC']
    
    res_df = res_df.sort_values(by = 'AIC', ascending = True).reset_index(drop = True)

    return res_df

# Create concat dataframe 
df_concat_diff = df_concat[['GBP/USD - Open', 'EUR/USD - Open']].diff()[1:]

# Train/Test split.
train = df_concat_diff[:92]
test = df_concat_diff[92:]

model_df = optimize_VAR(train)
print(model_df)

# Use the granger test.
print('EUR/USD - Open Granger-causes GBP/USD - Open? \n')
print('--------------------')
granger_GBP = grangercausalitytests(df_concat[['GBP/USD - Open', 'EUR/USD - Open']].diff()[1:], [4])

print('\nGBP/USD - Open Granger-causes EUR/USD - Open? \n')
print('--------------------')
granger_EUR = grangercausalitytests(df_concat[['EUR/USD - Open', 'GBP/USD - Open']].diff()[1:], [4])

