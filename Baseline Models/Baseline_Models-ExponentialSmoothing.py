# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:27:15 2022

Baseline study - Holt Winter’s Exponential Smoothing Model
#https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html

@author: Gonçalo Mateus
"""

#%% Imports

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from darts import TimeSeries

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm

from darts.models import ExponentialSmoothing

import warnings
warnings.filterwarnings("ignore")

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

#%% Functions to support the the model pre processing

def loadData(layer, timeStep):
    """
        Function that loads data, organize dates and return an dictionary 
        with the requested dataframes
        
        Return dataframe list of servers with the corresponding time step 
        
        layer - layer to analyze
        timeStep - time step
    """
    
    servers_entries = os.listdir('Data/'+layer+'/'+timeStep)
    list_servers = []
    
    for server in servers_entries:
        file = pd.read_csv('Data/'+layer+'/'+timeStep+'/'+server) 
        timeGraph = []        
        for time in file["timestamp"]:
            timeGraph.append(datetime.fromtimestamp(time))            
        file["date"]=timeGraph
        file["hour"] = file["date"].dt.hour
        file["just_date"] = file["date"].dt.date
        
        if 'Unnamed: 0' in file.columns:   
            file = file.drop(['Unnamed: 0'], axis=1)
        
        list_servers.append(file)

    return list_servers

def load_patterns_data(timeStep):
    """
        Function that loads patterns data, and return an dictionary 
        with the requested dataframes
        
        Return dataframe list of servers with the corresponding time step 
        
        timeStep - time step
    """
    
    servers_entries = os.listdir('Data/Patterns_files/'+timeStep)
    
    list_files = []
    
    for server in servers_entries:
        file = pd.read_csv('Data/Patterns_files/'+timeStep+'/'+server) 
        
        if 'Unnamed: 0' in file.columns:   
            file = file.drop(['Unnamed: 0'], axis=1)
        
        file['date'] = pd.to_datetime(file['date'], format='%Y-%m-%d %H:%M:%S')
        file['date'] = correctDates(file)
        list_files.append(file)

    predictdates = pd.read_csv('Data/Patterns_files/6_hours_toPredictDates.csv')
    predictdates['dates'] = pd.to_datetime(predictdates['dates'], format='%Y-%m-%d %H:%M:%S')
    predictdates['dates2'] = pd.to_datetime(predictdates['dates2'], format='%Y-%m-%d %H:%M:%S')

    return list_files, predictdates

def correctDates(file):
    """
        Function that correct some small errors in dates due due to time zone 
        changes
        
        Return a list with the correct dates
        
        file - file to correct dates
    """
    
    dateSize = len(file)
    firstDate = file.iloc[0]["date"] 
    newDates = []
    newDates.append(firstDate)
    
    for i in range(dateSize-1):
        firstDate = firstDate + timedelta(hours=6)    
        newDates.append(firstDate)    
        
    return newDates
    
def pre_process_statistics(file):
    """
        Pre process statistics - ACF and PACF
                
        file - dataframe 
    """
    
    # Stationarity
    ad_fuller_result = adfuller(file["CPU utilization (average)"])
    print(f'ADF Statistic: {ad_fuller_result[0]}')
    
    if(ad_fuller_result[0] < 0.05):
        print("Is stationary with",f'p-value: {ad_fuller_result[1]}')
    else:
        print("Is not stationary with",f'p-value: {ad_fuller_result[1]}')

    #ACF
    #plt.figure(figsize = [20,20])
    #plot_acf(file, lags = 40)
    #plt.show()
    
    #PACF
    #plt.figure(figsize = [20,20])
    #plot_pacf(file,  lags = 40)
    #plt.show() 
    

def get_dates_to_predict(dates_file, name):
    """
        Get dates to predict
        
        Return set of dates
        
        dates_file - file with dates
        name - name of the files that we want to obtain dates
    """
    dates_csv = dates_file[dates_file["name"] == name]
    if(dates_csv["trendsNum"].iloc[0] == 1):
        return [dates_csv["dates"].iloc[0]]
    else:
        return [dates_csv["dates"].iloc[0], dates_csv["dates2"].iloc[0]]

def split_dataframe(file, date_to_split, trainDays_range, name):
    """
        Create a TimeSeries, and split in train and validation sets
        
        Return a train and validation sets
        
        file - dataframe
        date_to_split - date to split sets
        trainDays_range - get x days to train
    """
    
    finalTrain = file[file["date"] == date_to_split].index[0]    
    finalValidation = file[file["date"] == date_to_split+datetime.timedelta(days=trainDays_range)].index[0]
    
    series = TimeSeries.from_dataframe(file, 'date', 'CPU utilization (average)', freq='6H')
    
    train = series[0:finalTrain]
    #train = train[-600:]
    
    val = series[finalTrain:finalValidation]  
    train_df = file[0:finalTrain]
    val_df = file[finalTrain:finalValidation]  
        
    return train, val, train_df, val_df


#%% Functions to support Prophet model
def run_exponential_smoothing_model(dataframe, seasonal_periods_val=None):
    """
        Return exponential smoothing model with gice definitions and fit 
        
        dataframe - dataframe to train
        trend_val - type of trend component.
        seasonal_val - type of seasonal component
        seasonal_periods_val - the number of periods in a complete seasonal 
                            cycle, e.g., 4 for quarterly data or 7 for daily 
                            data with a weekly cycle. If not set, inferred from 
                            frequency of the series. 
    """
    
    #Define the model
    model = ExponentialSmoothing(seasonal_periods=seasonal_periods_val)

    #Fit the model
    model.fit(dataframe)

    return model

def compute_errors(forecast, val, periods):
    """
        Compute model errors
        
        forecast - forecast
        val - validation set
        periods - periods to validate
    """
        
    #Compute Errors (MAE and MSE)
    mae_cross_val = mean_absolute_error(forecast, val[:periods])
    mse_cross_val = mean_squared_error(forecast, val[:periods])
    mape = mean_absolute_percentage_error(forecast, val[:periods])

    #print('MAE: ', mae_cross_val)
    #print('MSE: ', mse_cross_val)
    
    return mae_cross_val, mse_cross_val, mape

#%% Load files

#Load data of PL layer - 7 server + 1 average
pl_layer_6_hours = loadData("PL", "6_hours")
pl_layer_30_minutes = loadData("PL", "30_minutes")

#Load data of BL layer - 9 server + 1 average
bl_layer_6_hours = loadData("BL", "6_hours")
bl_layer_30_minutes = loadData("BL", "30_minutes")

#Load data of patterns - 6 hours
import datetime
patterns_data, predictDates_6hours = load_patterns_data("6_hours")

#%% INFO EXPONENCIAL SMOOTHING: Run Models - 6 hours step

#Data corresponds to patterns that exists along servers


#For guaranteeing that information from the future was not being considered 
#when making predictions on past values, each prediction was done by training 
#the model with data until the day prior to the prediction and predicting the 
#following day

#The MSE and Mean Absolute Error (MAE) were computed for each fold, and then 
#averaged to yield the final score for each model.


#%% EXPONENCIAL SMOOTHING 

names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_exponential_smoothing = []

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
    # 4: Standardize the data
    mean_y = server["CPU utilization (average)"].mean()
    std_y = server["CPU utilization (average)"].std()
    server["CPU utilization (average)"] = (server["CPU utilization (average)"]-mean_y)/std_y
    
    # 1: Get dates to split ranges
    dates_to_split = get_dates_to_predict(predictDates_6hours, names[count])
    date_count=0
    for date in dates_to_split:
        cross_validation_date = date
        
        mae_cross = []
        mse_cross = []
        mape_cross = []
        
        dataframe_results_pred = []
        dataframe_results_val = []
        dataframe_results_dates = []
        
        #30-times cross validation
        for x in tqdm(range(0, 29)):
            
            
            # 1.1: Get dataframe
            dataToAnalyze = server
            
            # 2: Split dataframe in train and validation set
            train_timeseries, val_timeseries, train_df, val_df = split_dataframe(dataToAnalyze, cross_validation_date, 1, names[count]+"--"+str(str(date).split(" ")[0]))
                 
            # 3: Statistics
            #dataToAnalyze = dataToAnalyze.set_index('date')
            #dataToAnalyze = dataToAnalyze.iloc[:,1:2]
            #pre_process_statistics(dataToAnalyze)
                        
            # 5: Get, define and fit the model
            seasonal = 28
            exponential_smoothing_model = run_exponential_smoothing_model(train_timeseries, seasonal)
            
            periods = 4        
            forecast = exponential_smoothing_model.predict(periods)
            
            y_pred = (forecast._xa.values.flatten())*std_y+mean_y
            dataframe_results_pred = np.concatenate((dataframe_results_pred, y_pred))     
            
            y_val = val_df["CPU utilization (average)"]*std_y+mean_y
            dataframe_results_val = np.concatenate((dataframe_results_val, y_val))      
            
            dates_aux_save = []
            for x in val_df["date"].values:
                dates_aux_save.append(str(x))
                
            dataframe_results_dates = np.concatenate((dataframe_results_dates, dates_aux_save))
            
            mae, mse, mape = compute_errors(y_pred, y_val, periods)
            mae_cross.append(mae)
            mse_cross.append(mse)           
            mape_cross.append(mape)
            
            cross_validation_date = cross_validation_date + datetime.timedelta(days=1)
            
        print('MAE_Cross: ', np.mean(mae_cross))
        print('MSE_Cross: ', np.mean(mse_cross))
        print('MAPE_Cross: ', np.mean(mape_cross))

        result_exponential_smoothing.append([names[count], np.mean(mae_cross), np.mean(mse_cross), np.mean(mape_cross)])
        
        aux_res = pd.DataFrame(dataframe_results_pred, columns=['y_pred'])
        aux_res['y_val'] = dataframe_results_val
        aux_res["date"] = dataframe_results_dates
        aux_res.to_csv("exponentialsmoothing_"+ str(count) + "_" + str(date_count)  +".csv")
        date_count = date_count +1
        print('# ================================================================')    
        # 7: With the best model test with cross validation
        
    count += 1
    
#result_exponential_smoothing = pd.DataFrame(result_exponential_smoothing)
#result_exponential_smoothing.columns = ['server', 'MAE', 'MSE', 'MAPE']
#Blocked_150
#result_exponential_smoothing.to_csv(f'Results/result_exponential_smoothing({seasonal})(Normal_split).csv')



# 8: Test with covariates


