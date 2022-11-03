# Ensemble Model (Interpretable)


1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past, future, and static co-variates

## Folder Structure

<pre>
<b>Ensemble Model (Interpretable)/</b>  
│  
├─── <b>EAMDrift_model/</b>  
│    ├─── Ensemble_Model_Class.py  
│    ├─── Models.py  
│    └─── <b>ModelsDB/</b>  
│         ├─── ExponentialSmoothingClass.py  
│         ├─── Prophet.py  
│         ├─── SARIMA.py  
│         ├─── LSTM.py  
│         └─── other_model.py  
│  
├─── Run Model.py  
│  
└─── README.md  
</pre>

### Files Descriptions

**Ensemble_Model_Class.py**

This script contains the code to apply anomaly detection methods to data from four sensors (water temperature, specific conductance, pH, dissolved oxygen) at six sites in the Logan River Observatory. 

**Models.py**

**Ensemble_Model_Class.py**

**ModelsDB files**

**Run Model.py**

## EAMDrift model

### Prerequisites:

This model depends on the following Python packages:

- darts
- river
- numpy
- pandas
- sklearn
- math
- typing
- tqdm
- datetime
- warnings

### EAMDrift:

<b>class EAMDriftModel(```timeseries_df_, columnToPredict_, time_column_, models_to_use_, dataTimeStep_,
                       trainning_samples_size_ = None,
                       trainning_points_ = 150,
                       prediction_points_ = 4,
                       to_extract_features_ = True```)</b>

<b>Parameters</b>
 - timeseries_df_(object) - Number of time steps to be input to the forecasting module.
 - columnToPredict_(str)
 - time_column_(str)
 - models_to_use_(list)
 - dataTimeStep_(str)
 - trainning_samples_size_(Optional[int])
 - trainning_points_(int)
 - prediction_points_(int)
 - to_extract_features_(bool)

#### Methods
[create_trainning_set()](#create_trainning_set) - Split dataframe in subsets to train the ensemble model

[fit()](#fit) - Fit/train the model on one series.

[historical_forecasts()](#historical_forecasts) - Compute the historical forecasts that would have been obtained by this model on the series.


##### <a name="create_trainning_set"></a> create_trainning_set()
##### <a name="fit"></a> fit()
Train the model with a specific darts.utils.data.TrainingDataset instance. These datasets implement a PyTorch Dataset, and specify how the target and covariates are sliced for training
**Returns:** Fitted model.
**Returns type:** self.

##### <a name="historical_forecasts"></a> historical_forecasts()

## Example Usage (EAMDrift model tutorial):

<pre>

</pre>






