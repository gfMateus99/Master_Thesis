# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:27:15 2022

Baseline study - Transformer Model
#https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

@author: Gonçalo Mateus
"""

#%% Imports

import pandas as pd
from tqdm import tqdm
import math
import itertools
import os
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mae, mse
from darts.models.forecasting.transformer_model import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

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
    
    series = TimeSeries.from_dataframe(file, 'date', 'CPU utilization (average)', freq='6H')[0:finalValidation]
        
    train = series[0:finalTrain]
    val = series[finalTrain:finalValidation]  
    train_df = file[0:finalTrain]
    val_df = file[finalTrain:finalValidation]  
    
    #train = train[-600:]
    #train_df = train_df[-600:]
    
    series = series[0:finalValidation]
    #series = series[-720:]
    
    plt.figure (figsize = (13, 7))
    train.plot()
    val.plot()
    plt.xlabel('Date', size=14)
    plt.ylabel("CPU utilization (average)", size=14)
    plt.title(name)
    plt.legend()
    plt.savefig('Plots/PredictGraphs/'+name+'.pdf', bbox_inches='tight')
    
    return train, val, series, train_df, val_df


#%% Functions to support TransformerModel model
def run_transformer_model(transformer, trainTransformed, valTransformed, series, date_split, 
                   model_="TransformerModel", dropout_=0, batch_size_=16, n_epochs_=300, 
                   lr=1e-5, model_name_="Lstm_model", random_state_=42, num_encoder_layers_=3, 
                   num_decoder_layers_=4, dim_feedforward_=512, 
                   input_chunk_length_=28, loss = "MSE", output_chunk_length_=4, 
                   activation_='relu',train=None, val=None, covariates_array=[], id_op = None):
    """
        Return TransformerModel model, mae_cross_val, mse_cross_val with given definitions
        
        transformer - trasnformer created to scale the data
        trainTransformed - data to be trained
        valTransformed - data to be validates
        series - series of all data 
        date_split - date to split train and validation sets
    """
        
    # 1: Create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(), freq=series.freq_str, periods=len(series)),
        attribute="year",
        one_hot=False,
    )
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True
    )
    covariates = year_series.stack(month_series)
    
    if (len(covariates_array)!=0):
        ram_series = pd.concat([train, val], ignore_index=True)
        ram_series = TimeSeries.from_dataframe(ram_series, 'date', covariates_array, freq='6H')  
        ram_series = Scaler().fit_transform(ram_series)
        covariates = covariates.stack(ram_series)

    cov_train, cov_val = covariates.split_before(date_split+datetime.timedelta(hours=18))

    ll = nn.MSELoss()
    if loss == "l1": 
        ll = nn.L1Loss()
        
    # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
    # a period of 5 epochs (`patience`)
    my_stopper = EarlyStopping(
        monitor="train_loss",
        patience=5,
        min_delta=0.05,
        mode='min',
    )
    
    # 2: Define the model   
    model = TransformerModel(
        input_chunk_length=input_chunk_length_, 
        output_chunk_length=output_chunk_length_, 
        d_model=64, 
        nhead=4, 
        optimizer_kwargs={"lr": lr},
        num_encoder_layers=num_encoder_layers_, 
        num_decoder_layers=num_decoder_layers_, 
        dim_feedforward=dim_feedforward_, 
        dropout=dropout_, 
        activation=activation_,
        loss_fn= ll,
        batch_size=batch_size_,
        n_epochs=n_epochs_,
        model_name=model_name_,
        random_state=random_state_,
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        pl_trainer_kwargs={"callbacks": [my_stopper],"accelerator": "cpu"}
    )
    
    # 3: Fit the model
    model.fit(
        trainTransformed,
        past_covariates=cov_train,
        val_past_covariates=cov_val,
        val_series=val_transformed,
        verbose=False
    )
    
    backtest_series = model.historical_forecasts(
        series,
        past_covariates=covariates,
        start=date_split,
        forecast_horizon=4,
        retrain=True,
        verbose=False,
    )

    plt.figure(figsize=(8, 5))
    series.plot(label="actual")
    backtest_series.plot(label="backtest")
    plt.legend()
    #plt.title("Backtest test")
    
    # 4: Compute errors
    mae_cross_val = mae(transformer.inverse_transform(series), transformer.inverse_transform(backtest_series))
    mse_cross_val = mse(transformer.inverse_transform(series), transformer.inverse_transform(backtest_series)) 
    
    mape_cross_val = mean_absolute_percentage_error(transformer.inverse_transform(series)._xa.values.flatten()[-120:], 
                                         transformer.inverse_transform(backtest_series)._xa.values.flatten()) 
    
    
    if (id_op != None):
        dates_aux_save = []
        for x in series._xa.date.values[-120:]:
            dates_aux_save.append(str(x))
            
        aux_res = pd.DataFrame(transformer.inverse_transform(backtest_series)._xa.values.flatten(), columns=['y_pred'])
        aux_res['y_val'] = transformer.inverse_transform(series)._xa.values.flatten()[-120:]
        aux_res["date"] = dates_aux_save
        aux_res.to_csv("transformer_"+ str(id_op)  +".csv")
        
    return model, mae_cross_val, mse_cross_val, mape_cross_val

def get_best_model(transformer, trainTransformed, valTransformed, seriesTransformed, date_split,parameters):
    """
        Function to find the best parameters for the lstm model
        
        transformer - trasnformer created to scale the data
        trainTransformed - data to be trained
        valTransformed - data to be validates
        series - series of all data 
        date_split - date to split train and validation sets
    """
    #dropout_=float(parameters[0]), 
    #lr=float(parameters[1]),
    #loss = parameters[2].split("'")[1],
    
    n_dropout= [0,0.15,0.3,0.5 ] 
    lr = [1e-3,1e-4,1e-5]
    loss = ["MSE", "l1"]
    encoder_layers=[3,4,5]
    decoder_layers=[3,4,5]
    feedforward=[512,1024,2048]
    combinations = [(x[0],x[1],x[2],x[3],x[4],x[5]) for x in list(itertools.product(n_dropout,lr,loss,encoder_layers,decoder_layers,feedforward))]
    
    best_mae = math.inf
    best_mse = math.inf
    best_specs = (0,0,0)
    best_mape = math.inf
    
    for x in tqdm(combinations):
        lstm_model, mae_cross_val, mse_cross_val, mape_cross_val = run_transformer_model(transformer, 
                                                                  trainTransformed, 
                                                                  valTransformed, 
                                                                  seriesTransformed, 
                                                                  date-datetime.timedelta(hours=18), 
                                                                  dropout_=x[0], 
                                                                  lr=x[1],
                                                                  loss = x[2],
                                                                  num_encoder_layers_=x[3], 
                                                                  num_decoder_layers_=x[4], 
                                                                  dim_feedforward_=x[5],
                                                                  activation_='relu',
                                                                  random_state_=42)
    
        if mae_cross_val < best_mae:
            best_mae = mae_cross_val
            best_mse = mse_cross_val
            best_specs = x
            best_mape = mape_cross_val
                
    return best_mae, best_mse, best_specs, best_mape

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

#%% INFO Transformer: Run Models - 6 hours step

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

#optimizer_cls – The PyTorch optimizer class to be used. Default: torch.optim.Adam.


#%% Transformer
"""
names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_transformer = []

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
    # 1: Get dates to split ranges
    dates_to_split = get_dates_to_predict(predictDates_6hours, names[count])
    
    for date in dates_to_split:
        
        cross_validation_date = date
                
        # 1.1: Get dataframe
        dataToAnalyze = server
        
        # 2: Split dataframe in train and validation set
        train_timeseries, val_timeseries, series_timeseries, train_df, val_df= split_dataframe(dataToAnalyze, cross_validation_date, 30, names[count]+"--"+str(str(date).split(" ")[0]))

        # 3: Statistics
        #dataToAnalyze = dataToAnalyze.set_index('date')
        #dataToAnalyze = dataToAnalyze.iloc[:,1:2]
        #pre_process_statistics(dataToAnalyze)
        
        # 4: Normalize the time series 
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train_timeseries)
        val_transformed = transformer.transform(val_timeseries)
        series_transformed = transformer.transform(series_timeseries)
        
        # 5: Get, define and fit the model
        best_mae_cross_val, best_mse_cross_val, specs, mape_cross_val = get_best_model(transformer, train_transformed, val_transformed, series_transformed, date, parameters)

        print('Model overview: ', specs)        
        print('MAE_Cross: ', best_mae_cross_val)
        print('MSE_Cross: ', best_mse_cross_val)
        
        result_transformer.append([names[count], specs, best_mae_cross_val, best_mse_cross_val])
        
        result_transformer_test = pd.DataFrame(result_transformer)
        result_transformer_test.columns = ['Pattern', 'Specs', 'MAE', 'MSE']
        result_transformer_test.to_csv(f'Results/result_transformer_test({names[count].split("(")[0]}_{str(str(date).split(" ")[0])}).csv')
        
        print('# ================================================================')    
        
    count += 1
    
    
result_transformer = pd.DataFrame(result_transformer)
result_transformer.columns = ['Pattern', 'Specs', 'MAE', 'MSE']
result_transformer.to_csv('Results/result_transformer_v1.csv')
"""
#%% Transformer

names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_transformer = []

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
    # 1: Get dates to split ranges
    dates_to_split = get_dates_to_predict(predictDates_6hours, names[count])
    date_count=0
    for date in dates_to_split:
        if(count>3):
            print(date)
            results_model = pd.read_csv(f'Results/result_transformer_backup/result_transformer_test({names[count].split("(")[0]}_{str(str(date).split(" ")[0])}).csv') 
            
            parameters = results_model.iloc[len(results_model)-1]["Specs"].split("(")[1].split(")")[0].split(",")
    
            cross_validation_date = date
                    
            # 1.1: Get dataframe
            dataToAnalyze = server
            
            # 2: Split dataframe in train and validation set
            train_timeseries, val_timeseries, series_timeseries, train_df, val_df= split_dataframe(dataToAnalyze, cross_validation_date, 30, names[count]+"--"+str(str(date).split(" ")[0]))
    
            # 3: Statistics
            #dataToAnalyze = dataToAnalyze.set_index('date')
            #dataToAnalyze = dataToAnalyze.iloc[:,1:2]
            #pre_process_statistics(dataToAnalyze)
            
            # 4: Normalize the time series 
            transformer = Scaler()
            train_transformed = transformer.fit_transform(train_timeseries)
            val_transformed = transformer.transform(val_timeseries)
            series_transformed = transformer.transform(series_timeseries)
            
            # 5: Get, define and fit the model
            #best_mae_cross_val, best_mse_cross_val, specs, mape_cross_val = get_best_model(transformer, train_transformed, val_transformed, series_transformed, date)
    
            lstm_model, mae_cross_val, mse_cross_val, mape_cross_val = run_transformer_model(transformer, 
                                                                      train_transformed, 
                                                                      val_transformed, 
                                                                      series_transformed, 
                                                                      date-datetime.timedelta(hours=18), 
                                                                      dropout_=float(parameters[0]), 
                                                                      lr=float(parameters[1]),
                                                                      loss = parameters[2].split("'")[1],
                                                                      num_encoder_layers_=int(parameters[3]), 
                                                                      num_decoder_layers_=int(parameters[4]), 
                                                                      dim_feedforward_=int(parameters[5]), 
                                                                      activation_='relu',
                                                                      random_state_=42,
                                                                      id_op =  str(count) + "_" + str(date_count))
    
            print('Model overview: ', parameters)        
            print('MAE_Cross: ', mae_cross_val)
            print('MSE_Cross: ', mse_cross_val)
            print('Mape_Cross: ', mape_cross_val)
           
            result_transformer.append([names[count], parameters, mae_cross_val, mse_cross_val, mape_cross_val])
            
            #result_transformer_test = pd.DataFrame(result_transformer)
            #result_transformer_test.columns = ['Pattern', 'Specs', 'MAE', 'MSE']
            #result_transformer_test.to_csv(f'Results/result_transformer_test({names[count].split("(")[0]}_{str(str(date).split(" ")[0])}).csv')
            date_count = date_count +1
            print('# ================================================================')    
            
    count += 1
    
#result_transformer = pd.DataFrame(result_transformer)
#result_transformer.columns = ['Pattern', 'Specs', 'MAE', 'MSE', 'MAPE']
#result_transformer.to_csv('Results/new_result_transformer(Normal_split).csv')


#%% 8: Test with covariates
"""
combinations = [['Write wait time'],['RAM used'],
 ['Round trip average'],
 ['Write wait time', 'RAM used'],
 ['Write wait time', 'Round trip average'],
 ['RAM used', 'Round trip average'],
 ['Write wait time', 'RAM used', 'Round trip average']]
    

names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_transformer = []

for server in patterns_data:
    print('# ================================================================')    
    print(names[count]) #Pattern + Server name
    
    # 1: Get dates to split ranges
    dates_to_split = get_dates_to_predict(predictDates_6hours, names[count])
    
    for date in dates_to_split:
        for comb in combinations:
            print(comb)
            results_model = pd.read_csv(f'Results/result_transformer_backup/result_transformer_test({names[count].split("(")[0]}_{str(str(date).split(" ")[0])}).csv') 
            
            parameters = results_model.iloc[len(results_model)-1]["Specs"].split("(")[1].split(")")[0].split(",")
    
            
            cross_validation_date = date
                    
            # 1.1: Get dataframe
            dataToAnalyze = server
            
            # 2: Split dataframe in train and validation set
            train_timeseries, val_timeseries, series_timeseries, train_df, val_df= split_dataframe(dataToAnalyze, cross_validation_date, 30, names[count]+"--"+str(str(date).split(" ")[0]))
    
            # 3: Statistics
            #dataToAnalyze = dataToAnalyze.set_index('date')
            #dataToAnalyze = dataToAnalyze.iloc[:,1:2]
            #pre_process_statistics(dataToAnalyze)
            
            # 4: Normalize the time series 
            transformer = Scaler()
            train_transformed = transformer.fit_transform(train_timeseries)
            val_transformed = transformer.transform(val_timeseries)
            series_transformed = transformer.transform(series_timeseries)
            
            # 5: Get, define and fit the model
            #best_mae_cross_val, best_mse_cross_val, specs, mape_cross_val = get_best_model(transformer, train_transformed, val_transformed, series_transformed, date)
    
            lstm_model, mae_cross_val, mse_cross_val, mape_cross_val = run_transformer_model(transformer, 
                                                                      train_transformed, 
                                                                      val_transformed, 
                                                                      series_transformed, 
                                                                      date-datetime.timedelta(hours=18), 
                                                                      dropout_=float(parameters[0]), 
                                                                      lr=float(parameters[1]),
                                                                      loss = parameters[2].split("'")[1],
                                                                      activation_='relu',
                                                                      random_state_=42,
                                                                      train = train_df, 
                                                                      val = val_df,
                                                                      covariates_array=comb)
    
            print('Model overview: ', parameters)        
            print('MAE_Cross: ', mae_cross_val)
            print('MSE_Cross: ', mse_cross_val)
            print('Mape_Cross: ', mape_cross_val)
           
            result_transformer.append([names[count], comb, mae_cross_val, mse_cross_val, mape_cross_val])
            
            #result_transformer_test = pd.DataFrame(result_transformer)
            #result_transformer_test.columns = ['Pattern', 'Specs', 'MAE', 'MSE']
            #result_transformer_test.to_csv(f'Results/result_transformer_test({names[count].split("(")[0]}_{str(str(date).split(" ")[0])}).csv')
            
            print('# ================================================================')    
        
    count += 1
    
result_transformer = pd.DataFrame(result_transformer)
result_transformer.columns = ['Pattern', 'Specs', 'MAE', 'MSE', 'MAPE']
#result_transformer.to_csv('Results/result_transformer(all_combinations).csv')


"""






