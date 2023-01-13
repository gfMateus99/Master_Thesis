# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:27:15 2022

Baseline study - LSTM Model
#https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html?highlight=rnnmodel#darts.models.forecasting.rnn_model.RNNModel

@author: Gonçalo Mateus
"""

#%% Imports

import pandas as pd
from tqdm import tqdm
import math
import itertools
from datetime import datetime, timedelta
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import mae, mse, mape

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


#%% Functions to support LSTM model
def run_lstm_model(transformer, trainTransformed, valTransformed, series, train_df, date_split, model_="LSTM", hidden_dim_=20, 
                   dropout_=0, batch_size_=16, n_epochs_=300, lr=1e-5, n_rnn_layers_=1, model_name_="Lstm_model", 
                   random_state_=42, training_length_=32, input_chunk_length_=28, loss = "MSE", id_op = None):
    """
        Return lstm model, mae_cross_val, mse_cross_val with given definitions
        
        transformer - trasnformer created to scale the data
        trainTransformed - data to be trained
        valTransformed - data to be validates
        series - series of all data 
        date_split - date to split train and validation sets
    """
        
    # 1: Create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1600),
        attribute="year",
        one_hot=False,
    )
    
    year_series = Scaler().fit_transform(year_series)
    month_series = datetime_attribute_timeseries(
        year_series, attribute="month", one_hot=True
    )
    
    covariates = year_series.stack(month_series)
    cov_train, cov_val = covariates.split_before(date_split)
    
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

    my_stopper2 = EarlyStopping(
        monitor="val_loss",
        patience=5,
        min_delta=0.05,
        mode='min',
    )

    # 2: Define the model
    model = RNNModel(
        model=model_,
        hidden_dim=hidden_dim_,
        dropout=dropout_,
        batch_size=batch_size_,
        n_epochs=n_epochs_,
        optimizer_kwargs={"lr": lr},
        model_name=model_name_,
        loss_fn= ll,
        log_tensorboard=True,
        random_state=random_state_,
        training_length=training_length_,
        input_chunk_length=input_chunk_length_,
        n_rnn_layers=n_rnn_layers_,
        force_reset=True,
        save_checkpoints=True,
        #pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True}
        pl_trainer_kwargs={"callbacks": [my_stopper, my_stopper2]}
    )
        
    # 3: Fit the model
    model.fit(
        trainTransformed,
        future_covariates=covariates,
        val_series=val_transformed,
        val_future_covariates=covariates,
        #past_covariates=ram_used,
        verbose=False
    )
    
    backtest_series = model.historical_forecasts(
        series,
        future_covariates=covariates,
        start=date_split,
        forecast_horizon=6,
        retrain=False,
        verbose=False,
    )
    """
    plt.figure(figsize=(8, 5))
    series.plot(label="actual")
    backtest_series.plot(label="backtest")
    plt.legend()
    plt.show()
    plt.title("Backtest, starting Jan 1959, 6-months horizon")
    """
    # 4: Compute errors
    mae_cross_val = mae(transformer.inverse_transform(series), transformer.inverse_transform(backtest_series))
    mse_cross_val = mse(transformer.inverse_transform(series), transformer.inverse_transform(backtest_series)) 
    mape_cross_val = mape(transformer.inverse_transform(series), transformer.inverse_transform(backtest_series)) 
    
    if (id_op != None):
        dates_aux_save = []
        for x in series._xa.date.values[-120:]:
            dates_aux_save.append(str(x))
            
        aux_res = pd.DataFrame(transformer.inverse_transform(backtest_series)._xa.values.flatten(), columns=['y_pred'])
        aux_res['y_val'] = transformer.inverse_transform(series)._xa.values.flatten()[-120:]
        aux_res["date"] = dates_aux_save
        aux_res.to_csv("lstm_"+ str(id_op)  +".csv")
    
    return model, mae_cross_val, mse_cross_val, mape_cross_val

def get_best_model(transformer, trainTransformed, valTransformed, seriesTransformed, train_df, date_split):
    """
        Function to find the best parameters for the lstm model
        
        transformer - trasnformer created to scale the data
        trainTransformed - data to be trained
        valTransformed - data to be validates
        series - series of all data 
        date_split - date to split train and validation sets
    """
    n_layers = range(1, 5)
    n_dropout= [0,0.15,0.3] 
    hid_dim = [5,10,20,50,100]
    lr = [1e-3,1e-4,1e-5]
    
    n_layers = range(1, 3)
    n_dropout= [0,0.15, 0.3] 
    hid_dim = [5,10,50,100]
    lr = [1e-3,1e-5]
    
    loss = ["MSE", "l1"]
    combinations = [(x[0],x[1],x[2],x[3],x[4]) for x in list(itertools.product(n_layers,n_dropout,hid_dim,lr,loss))]
    
    best_mae = math.inf
    best_mse = math.inf
    best_mape = math.inf
    best_specs = (0,0,0,0,0)
    
    for x in tqdm(combinations):
        lstm_model, mae_cross_val, mse_cross_val, mape_cross_val = run_lstm_model(transformer,
                                                                  trainTransformed, 
                                                                  valTransformed, 
                                                                  seriesTransformed, 
                                                                  train_df,
                                                                  date_split,
                                                                  n_rnn_layers_=x[0], 
                                                                  dropout_=x[1], 
                                                                  hidden_dim_=x[2], 
                                                                  lr=x[3],
                                                                  loss=x[4])
 
        if mae_cross_val < best_mae:
            best_mae = mae_cross_val
            best_mse = mse_cross_val
            best_mape = mape_cross_val
            best_specs = x
                
    return best_mae, best_mse, best_specs, best_mape

#%% Load files and add dates - dataset is already in hours

from datetime import datetime

data = pd.read_csv(r'google_cluster_data_organized.csv')
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")


train_df = data[0:223]

series_timeseries = TimeSeries.from_dataframe(data, 'date', 'cpu_usage', freq='1H')

        
train_timeseries = series_timeseries[0:223]
val_timeseries = series_timeseries[223:]  

transformer = Scaler()
train_transformed = transformer.fit_transform(train_timeseries)
val_transformed = transformer.transform(val_timeseries)
series_transformed = transformer.transform(series_timeseries)

#names = os.listdir('Data/Patterns_files/6_hours')
count = 0
result_lstm = []

best_mae_cross_val, best_mse_cross_val, specs, best_mape = get_best_model(transformer, train_transformed, val_transformed, series_transformed, train_df, data["date"][223])

