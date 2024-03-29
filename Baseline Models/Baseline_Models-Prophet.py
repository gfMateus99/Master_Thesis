# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:27:15 2022

Baseline study - Facebook Prophet MODEL
#https://facebook.github.io/prophet/docs/quick_start.html

@author: Gonçalo Mateus
"""

#%% Imports

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

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
    
    train_df = file[0:finalTrain]
    #train_df = train_df[-600:]
    
    val_df = file[finalTrain:finalValidation]
    
    return train_df, val_df


#%% Functions to support Prophet model

def run_prophet_model(prophet_dataframe, yearly_seasonality_val, daily_seasonality_val, 
                      weekly_seasonality_val, growth_val, seasonality_mode_val, holidays_val):
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
    model = Prophet(yearly_seasonality = True,
                    daily_seasonality = True,
                    weekly_seasonality = True,
                    growth = 'linear',
                    seasonality_mode = "multiplicative",
                   )
    #Fit the model
    model.fit(prophet_dataframe)

    return model

def compute_errors(forecast, val):
    """
        Compute model errors
        
        forecast - forecast
        val - validation set
    """
        
    #Compute Errors (MAE and MSE)
    
    mae_cross_val = mean_absolute_error(forecast, val)
    mse_cross_val = mean_squared_error(forecast, val)
    mape = mean_absolute_percentage_error(forecast, val)

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

#%% INFO PROPHET: Run Models - 6 hours step

#Data corresponds to patterns that exists along servers

#For guaranteeing that information from the future was not being considered 
#when making predictions on past values, each prediction was done by training 
#the model with data until the day prior to the prediction and predicting the 
#following day

#The MSE and Mean Absolute Error (MAE) were computed for each fold, and then 
#averaged to yield the final score for each model.


#%% PROPHET 

names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_prophet = []

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
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
        
        toplot_forecast = []
        
        train_df_test, val_df_test = split_dataframe(server, cross_validation_date, 30, names[count]+"--"+str(str(date).split(" ")[0]))
        
        #30-times cross validation
        for x in tqdm(range(0, 30)):
            
            # 1.1: Get dataframe
            dataToAnalyze = server
            
            # 2: Split dataframe in train and validation set
            train_df, val_df = split_dataframe(dataToAnalyze, cross_validation_date, 1, names[count]+"--"+str(str(date).split(" ")[0]))
                 
            train_df = train_df[["date", "CPU utilization (average)"]]
            train_df.columns = ['ds', 'y']
            
            val_df = val_df[["date", "CPU utilization (average)"]]
            val_df.columns = ['ds', 'y']
            
            # 3: Statistics
            #dataToAnalyze = dataToAnalyze.set_index('date')
            #dataToAnalyze = dataToAnalyze.iloc[:,1:2]
            #pre_process_statistics(dataToAnalyze)
            
            # 4: Standardize the data
            mean_y = train_df["y"].mean()
            std_y = train_df["y"].std()
            train_df["y"] = (train_df["y"]-mean_y)/std_y
            
            # 5: Get, define and fit the model
            prophet_model = run_prophet_model(train_df, True, True, 
                                  True, "linear", "multiplicative", "")
            
            periods = 4        
            future = prophet_model.make_future_dataframe(periods=periods, freq="6H")
            forecast = prophet_model.predict(future)
            
            y_pred = forecast['yhat'].values*std_y+mean_y
            y_pred = pd.DataFrame(y_pred)
            y_pred = y_pred[-periods:][0].values
            toplot_forecast = np.concatenate((toplot_forecast, y_pred))

            dates_aux_save = []
            for y in val_df["ds"].values:
                dates_aux_save.append(str(y))
                
            dataframe_results_dates = np.concatenate((dataframe_results_dates, dates_aux_save))
            
            dataframe_results_pred = np.concatenate((dataframe_results_pred, y_pred))

            dataframe_results_val = np.concatenate((dataframe_results_val, val_df['y'][:periods])) 

            mae, mse, mape = compute_errors(y_pred, val_df['y'][:periods])
            mae_cross.append(mae)
            mse_cross.append(mse)
            mape_cross.append(mape)
           
            
            cross_validation_date = cross_validation_date + datetime.timedelta(days=1)
            #print(cross_validation_date)
            
        print('MAE_Cross: ', np.mean(mae_cross))
        print('MSE_Cross: ', np.mean(mse_cross))
        print('MAPE_Cross: ', np.mean(mape_cross))

        result_prophet.append([names[count], np.mean(mae_cross), np.mean(mse_cross),  np.mean(mape_cross)])
         
        plt.figure (figsize = (20, 7))
        
        #ax = sns.lineplot(y="CPU utilization (average)", x="date", data=train_df_test, color="blue")
        ax = sns.lineplot(y="CPU utilization (average)", x="date", data=val_df_test, color="blue")
        ax = sns.lineplot(y=toplot_forecast, x="date", data=val_df_test, color="orange")
        ax.set_title(names[count]+"--"+str(str(date).split(" ")[0]), size=18)
        plt.xticks(size=13)
        plt.yticks(size=13)
        plt.xlabel('Date', size=14)
        plt.ylabel('CPU utilization (average)', size=14)
        plt.show()
        
        aux_res = pd.DataFrame(dataframe_results_pred, columns=['y_pred'])
        aux_res['y_val'] = dataframe_results_val
        aux_res["date"] = dataframe_results_dates
        aux_res.to_csv("prophet_"+ str(count) + "_" + str(date_count)  +".csv")
        date_count = date_count +1
           
        print('# ================================================================')    
        
    count += 1
    
#result_prophet = pd.DataFrame(result_prophet)
#result_prophet.columns = ['server', 'MAE', 'MSE', 'MAPE']
#result_prophet.to_csv('Results/result_prophet_aux(150).csv')

# 8: Test with covariates

