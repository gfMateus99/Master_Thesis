# Ensemble Model (Interpretable)

## Preamble

1) Ensemble
2) Based on Mixure of Experts (weights)
3) Interpretable model
4) Real-time re-trainning
5) Allows past, future, and static co-variates

## Folder Structure

<pre>
Ensemble Model (Interpretable)/  
│  
├─── EAMDrift_model/  
│    ├─── Ensemble_Model_Class.py  
│    ├─── Models.py  
│    ├─── ModelsDB/  
│         ├─── ExponentialSmoothingClass.py  
│         ├─── Prophet.py  
│         ├─── SARIMA.py  
│         ├─── LSTM.py  
│         ├─── other_model.py  
│  
├─── Run Model.py  
│  
├─── README.md  
</pre>

## EAMDrift model

### Getting Started:
<pre>
1
</pre>

### Prerequisites:
<pre>
1
</pre>

### Methods and Parameters:

<pre>

<b>class darts.models.forecasting.arima.ARIMA(p=12, d=1, q=0, seasonal_order=(0, 0, 0, 0), trend=None, random_state=0)</b>

<b>Parameters</b>
 - p (int) – Order (number of time lags) of the autoregressive model (AR).
 - d (int) – The order of differentiation; i.e., the number of times the data have had past values subtracted (I).

  q (int) – The size of the moving average window (MA).

  seasonal_order (Tuple[int, int, int, int]) – The (P,D,Q,s) order of the seasonal component for the AR parameters, differences, MA parameters and periodicity.

  trend (str) – Parameter controlling the deterministic trend. ‘n’ indicates no trend, ‘c’ a constant term, ‘t’ linear trend in time, and ‘ct’ includes both. Default is ‘c’ for models without integration, and no trend for models with integration.
</pre>


### Input:

### Output:


## Example Usage (EAMDrift model):







