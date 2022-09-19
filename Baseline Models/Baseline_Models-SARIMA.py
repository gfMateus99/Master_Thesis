# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:27:15 2022

Baseline study - SARIMA MODEL
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html?highlight=sarima#statsmodels.tsa.statespace.sarimax.SARIMAX

@author: Gonçalo Mateus
"""

#%% Imports

import os
import pandas as pd
from datetime import datetime, timedelta

from darts import TimeSeries
import sklearn.metrics as sm
import statsmodels.api as stm
from matplotlib import pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from itertools import product
from darts.models import ARIMA
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


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
    val = series[finalTrain:finalValidation]  
    train_df = file[0:finalTrain]
    val_df = file[finalTrain:finalValidation]  
    
    plt.figure (figsize = (13, 7))
    train.plot()
    val.plot()
    plt.xlabel('Date', size=14)
    plt.ylabel("CPU utilization (average)", size=14)
    plt.title(name)
    plt.legend()
    plt.savefig('Plots/PredictGraphs/'+name+'.pdf', bbox_inches='tight')
    
    return train, val, train_df, val_df

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

def best_model(series,train, val, p,d,q):
    """

    """
    model = ARIMA(p,d,q)
    model.fit(train)
    prediction = model.predict(len(val))
    # series.plot()
    # prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
    # plt.legend()
    predictions = prediction._xa.values.flatten()
    val_predictions = val._xa.values.flatten()
    error = sm.mean_absolute_error(val_predictions, predictions)
    return error

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

#%% INFO SARIMA: Run Models - 6 hours step

#Data corresponds to patterns that exists along servers

#All the models were tested with different sets of covariates. These included: 
#   -Write wait time
#   -Filesystem Shrinking
#   -Input Bandwidth
#   -Output Bandwidth, 
#   -Input non-unicast packets
#   -RAM used
#   -Round trip average.

#For guaranteeing that information from the future was not being considered 
#when making predictions on past values, each prediction was done by training 
#the model with data until the day prior to the prediction and predicting the 
#following day

#The MSE and Mean Absolute Error (MAE) were computed for each fold, and then 
#averaged to yield the final score for each model.

#SARIMA 
#s - S
#p - AR
#d - I
#q - MA

#A problem with ARIMA is that it does not support seasonal data. That is a time 
#series with a repeating cycle. Clearly we have this kind of data

#Precondition of SARIMA: Stationarity
#ACF and PACF assume stationarity of the underlying time series.

#If the p-value is very less than the significance level of 0.05 and hence we 
#can reject the null hypothesis and take that the series is stationary. This is 
#tested in pre_process_statistics fucntion. 

#Two approaches were taken to determine the ideal SARIMA parameters: ACF and 
#PACF plots, and a grid search. The ACF and PACF plots were used as a starting 
#point to narrow down to a few potential parameters, and then a grid search was 
#used to identify the best parameters. 

#ACF - estimates the relative amount of information that will be lost by using 
#the model to represent the process. The less information a model loses, the 
#higher the quality of that model. The best model will be the one with the 
#lowest AIC value.

#PACF - 

#%% SARIMA 

names = os.listdir('Data/Patterns_files/6_hours')
count = 0

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
    # 1: Get dataframe and dates to split ranges
    dates_to_split = get_dates_to_predict(predictDates_6hours, names[count])
    for date in dates_to_split:
        
        dataToAnalyze = server
        # 2: Split dataframe in train and validation set
        train_timeseries, val_timeseries, train_df, val_df = split_dataframe(dataToAnalyze, date, 30, names[count]+"--"+str(str(date).split(" ")[0]))
                
        # 3: Statistics
        dataToAnalyze = dataToAnalyze.set_index('date')
        dataToAnalyze = dataToAnalyze.iloc[:,1:2]
        pre_process_statistics(dataToAnalyze)
        
        # 4: Standardize the data
        #mean_y = train_timeseries._xa.values.flatten().mean()
        #std_y = train_timeseries._xa.values.flatten().std()
        train_timeseries = train_timeseries._xa.values.flatten()
        #train_timeseries = (train_timeseries-mean_y)/std_y
        
        # 5: Define parameters
        p = range(0, 5, 1)
        d = range(0, 1, 1)
        q = range(0, 4, 1)
        
        P = range(0, 5, 1)
        D = range(0, 1, 1)
        Q = range(0, 4, 1)
        
        s = [4, 28] 
        # 6 hours * 4 = 24 hours - daily seasonality (4)
        # 7 * 6 hours * 4 = 7 days - weekly seasonality (28)

        parameters = product(p, d, q, P, D, Q, s)
        parameters_list = list(parameters)
        
        print(len(parameters_list))
        
        # 6: Tunning - get an dataframe with parameters, corresponding AIC and BIC
        result_sarima = get_best_model(parameters_list, s, train_timeseries)
        print(result_sarima.iloc[0])
        result_sarima.to_csv('Results/result_sarima_statistics/'+names[count]+"--"+str(str(date).split(" ")[0])+".csv" )
        print('# ================================================================')
        
        # 7: With the best model test with cross validation
        
        
        
    count += 1
    

# 8: Test with covariates

