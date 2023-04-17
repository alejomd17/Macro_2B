import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score)

def scalate_data(data, scale, type, scaler=None):
    """
    Function to scalate an descalate data by type of scala and
    type of work
    inputs:
        data : pd.series-> dataset to be scalate
        scale : string-> scala to work with it
        type : string-> If de data need to be scaled of descaled
    output:
        scaler-> object that can be sk object or just a number
        data_to_use : dataframe-> scaled or descaled data
    """
    sk_scales = {
            "acum_scal_01": MinMaxScaler(),
            "acum_scal_11": StandardScaler()
            }

    if type == "scalate":

        if scale in ("acum_scal_01", "acum_scal_11"):
            scaler = sk_scales[scale]
            scaler.fit(np.array(data).reshape(-1, 1))
            data_transform = scaler.transform(np.array(data).reshape(-1, 1)).reshape(-1)
        elif scale == "acum_scal_lg":
            temp_data = pd.DataFrame()
            scaler = abs(data.min()) + 1
            temp_data["acum_scal_lg"] = data + (scaler)
            data_transform = np.log(temp_data["acum_scal_lg"])  
        else:
            data_transform = data

        data_to_use = pd.DataFrame()
        data_to_use[scale] = data_transform

    else:
        
        if scale in ("acum_scal_01", "acum_scal_11"):
            data_transform = scaler.inverse_transform(np.array(data).reshape(-1, 1)).reshape(-1)
        elif scale == "acum_scal_lg":
            pred = np.array(list(data))
            pred = np.exp(pred)
            data_transform = pred - scaler
        else:
            data_transform = data

        data_to_use = pd.DataFrame()
        data_to_use["pred"] = data_transform

    return scaler, data_to_use

def select_best_model(trained_models):
    """
    Funtion which allows select the best model based on their metrics: rmse, mape, r2
    inputs: 
        trained_models : list-> tranined and evaluated models object list 
    outputs: 
        winner_model : tranined and evaluated winnwe model
    """

    winner_model = trained_models[0]

    for current in trained_models[1:]:
        contador = 0

        if current['metrics']["rmse"] < winner_model['metrics']["rmse"]: 
            contador += 1 
        if current['metrics']["mape"] < winner_model['metrics']["mape"]:
            contador += 1 
        if current['metrics']["r2"] > winner_model['metrics']["r2"]:
            contador += 1

        if contador >= 2:
            winner_model = current
    return winner_model

def evaluate_metrics(data_test, pred):

    rmse = np.sqrt(mean_squared_error(data_test, pred))
    mape = mean_absolute_percentage_error(data_test, pred)
    r_2 = r2_score(data_test, pred)

    metrics = {"rmse": rmse, "mape": mape, "r2": r_2}

    # print("test: ", data_test)
    # print("pred: ", pred)
    # print(metrics)
    return metrics

def obtain_steps(forecaster):
    current_year = datetime.now().year
    current_month = datetime.now().month - 1
    last_year_prev = forecaster.training_range[1].year
    last_month_prev = forecaster.training_range[1].month
    
    final_month = 12
    steps_final = final_month - current_month 

    diff_years = (current_year-last_year_prev)*12
    steps_pred = final_month - last_month_prev + diff_years

    list_year_month = []
    for i in range(steps_final):
        current_month = current_month + 1
        year_month = f"{current_year}-{current_month}"
        list_year_month.append(year_month)

    return steps_pred, steps_final, list_year_month



def series_to_supervised(data, n_lags=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: DataFrame with Time series.
        n_lags: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    var_name=list(data.columns)   
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_lags, 0, -1):
        cols.append(df.shift(i))
        names += [('%s_lag_%d' % (n, i)) for n in var_name]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s' % (n)) for n in var_name]
        else:
            names += [('%s_t+%d' % (n, i)) for n in var_name]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg