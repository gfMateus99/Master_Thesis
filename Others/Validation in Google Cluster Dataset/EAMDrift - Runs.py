# -*- coding: utf-8 -*-
"""

@author: Gonçalo Mateus

EAMDrift
----------

"""

# %% Imports

##############################################################################
# Imports
##############################################################################

from EAMDrift_model import EAMDriftModel
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# %% Load files

##############################################################################
# Load files
##############################################################################

data = pd.read_csv(r'google_cluster_data_organized.csv')
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M:%S")

covariates = data.copy()

covariates_col = covariates.columns
covariates = covariates.reset_index()
covariates.drop(covariates_col, axis = 1, inplace=True)
covariates_aux = covariates.copy()

# %% RUN MODEL

##############################################################################
# Model run
##############################################################################

if __name__ == '__main__':
    
    dataframe = data[["date", "cpu_usage"]].copy()

    models = ["TRANSFORMER", "LSTM", "LSTM2", "SARIMAX", "ExponentialSmoothing", "Prophet"]

    mean_y = dataframe["cpu_usage"].mean()
    std_y = dataframe["cpu_usage"].std()
    dataframe["cpu_usage"] = (dataframe["cpu_usage"]-mean_y)/std_y

    index = 223
    points_to_predict = 6
    ensemble_model = EAMDriftModel(timeseries_df_=dataframe[0:index],
                                   columnToPredict_="cpu_usage",
                                   time_column_="date",
                                   models_to_use_=models,
                                   dataTimeStep_="1H",
                                   covariates_=covariates_aux[0:index], #vamos assumir que ja bem bonitas aqui
                                   categorical_covariates_ = [], #["author_name_toRule", "topic_in_graph_toRule"],
                                   covariates_ma_ = 7*2, 
                                   error_metric_="MAPE",
                                   #trainning_samples_size_ = 100,
                                   trainning_points_=100, 
                                   fit_points_size_ = 100,
                                   prediction_points_=points_to_predict,
                                   to_extract_features_=True,
                                   use_dates_=True,
                                   #selectBest_=1,# None selects all
                                   to_retrain_ = True,
                                   n_jobs_=6
                                   )
    
    # Make Trainning set
    trainning_set_init, self_ = ensemble_model.create_trainning_set()     
    
    # Train ensemble method
    rules = ensemble_model.fitEnsemble() 
        
    # Predict with historical forecasts
    forecast, dates, y_true = ensemble_model.historical_forecasts(dataframe[index:], 
                                                                  forecast_horizon = int((len(dataframe)-index)/points_to_predict), 
                                                                  covariates = covariates_aux[index:])
        
    #SVC
    #(6.684294118899815, 124.3734986507135, 0.2818512370590435)

    #Rules
    #(6.721873162305343, 123.41092081767279, 0.27796812068439697)
    
    #RDM
    #(6.3357430429652455, 98.47008179017304, 0.2732373686643767)
        