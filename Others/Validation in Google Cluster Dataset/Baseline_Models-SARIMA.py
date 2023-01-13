# -*- coding: utf-8 -*-
"""

Validation study - SARIMA MODEL
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html?highlight=sarima#statsmodels.tsa.statespace.sarimax.SARIMAX

@author: Gonçalo Mateus
"""

#%% Imports

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from darts import TimeSeries

import sklearn.metrics as sm
import statsmodels.api as stm

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from itertools import product

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


#%% Functions to support SARIMA model

def get_best_model(parameters_list, s, train):
    """
        Return dataframe with parameters, corresponding AIC and BIC
        
        parameters_list - list with (p, d, q, P, D, Q) tuples
        s - length of season
        train - the train variable
    """
    results = []
    
    for param in tqdm(parameters_list):
        try: 
            model = stm.tsa.statespace.SARIMAX(train, order=(param[0], param[1], param[2]), seasonal_order=(param[3], param[4], param[5], param[6])).fit(disp=-1)
        except:
            continue   
   
        aic = model.aic
        bic = model.bic
        results.append([param, aic, bic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)x(P, D, Q)', 'AIC', 'BIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

def best_model(series,train, val, p, d, q, P, D, Q, s):
    """

    """
    model = stm.tsa.statespace.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    
    model.fit(train)
    
    prediction = model.predict(len(val))
    predictions = prediction._xa.values.flatten()
    val_predictions = val._xa.values.flatten()
    error = sm.mean_absolute_error(val_predictions, predictions)
    return error


def run_sarima_model(train, val, p, d, q, P, D, Q, s):
    """
        Return prophet model with gice definitions and fit 
        
        prophet_dataframe - dataframe to train
        yearly_seasonality - yearly seasonality
        daily_seasonality - daily seasonality
        weekly_seasonality - weekly seasonality
        growth - growth of model
        seasonality_mode - mode of seasonality
        holidays - holidays to model
    """
    #Define the model
    model = stm.tsa.statespace.SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)

    #Fit the model
    #model.fit(train)

    return model


def compute_errors(forecast, val, periods):
    """
        Compute model errors
        
        forecast - forecast
        val - validation set
        periods - periods to validate
    """
        
    #Compute Errors (MAE and MSE)
    
    mae_cross_val = mean_absolute_error(val, forecast)
    mse_cross_val = mean_squared_error(val, forecast)
    mape = mean_absolute_percentage_error(val, forecast)
    
    return mae_cross_val, mse_cross_val, mape


#%% Load files and add dates - dataset is already in hours

from datetime import datetime

data = pd.read_csv(r'google_cluster_data_organized.csv')
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")
 
#%% SARIMA

train_df = data[0:223]

series_timeseries = TimeSeries.from_dataframe(data, 'date', 'cpu_usage', freq='1H')
        
train_timeseries = series_timeseries[0:223]
val_timeseries = series_timeseries[223:]  

count = 0
result_lstm = []


train_timeseries = train_timeseries._xa.values.flatten()
series_timeseries = series_timeseries._xa.values.flatten()

# 5: Define parameters
p = range(0, 5, 1)
d = range(0, 1, 1)
q = range(0, 4, 1)

P = range(0, 5, 1)
D = range(0, 1, 1)
Q = range(0, 4, 1)

s = [24] 

parameters = product(p, d, q, P, D, Q, s)
parameters_list = list(parameters)

"""
0	(p, d, q)x(P, D, Q)	AIC	BIC
0	(4, 0, 3, 4, 0, 3, 24)	5608.746678109958	5654.867088458289
"""

mean_y = series_timeseries.mean()
std_y = series_timeseries.std()
series_timeseries = (series_timeseries-mean_y)/std_y

forecasts = []
for x in tqdm(range(223, len(data), 6)):

    train_data = series_timeseries[0:x]
    
    model = stm.tsa.statespace.SARIMAX(train_data, order=(4,0,3), seasonal_order=(4,0,3,24)).fit(disp=-1)
    forecast = model.predict(start=train_data.shape[0], end=train_data.shape[0]+5)
    
    forecasts = np.concatenate([forecasts, forecast])
    
#print(compute_errors(forecasts[:521]*std_y+mean_y, series_timeseries[223:]*std_y+mean_y, 24))


    















