
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import math
import pandas as pd

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from .auxiliar_functions import evaluate_metrics,series_to_supervised

class XgboostRegressorModel():

    def __init__(self, seed=123): 
        """
        atributes: 
            seed : int -> seed  
        """
        self.name ='XgboostRegressorModel'
        self.scaler_X =None
        self.scaler_y =None
        self.seed = seed
 
    def split_data(self, data, features, div):
        df_train = data[:-div]
        df_test = data[-div:]

        data = { "data": data[features],
                "train": df_train[features],
                "test": df_test[features],
                "features": features}
        return data

    def train_model(self, df_data, steps, y_name='twobe', metric_param='neg_root_mean_squared_error',scaler=None, n_folds=3):

        #create lags dataset
        #df_lags=series_to_supervised(df_data[features], n_lags=lags, n_out=1, dropnan=True)
        #df=df_lags[df_lags.columns[:-len(features)+1]]
        #df['mes']=df.index.month

        #scale features 
        if scaler!= None: 
            self.scaler_X=scaler().fit(df_data.drop(y_name,axis=1))
            self.scaler_y=scaler().fit(df_data[[y_name]])
            X_train=pd.DataFrame(self.scaler_X.transform(df_data.drop(y_name,axis=1)),columns=self.scaler_X.feature_names_in_,index=df_data.index)
            y_train=pd.DataFrame(self.scaler_y.transform(df_data[[y_name]]),columns=self.scaler_y.feature_names_in_,index=df_data.index)
        else:
            self.scaler_X=None
            self.scaler_y=None
            X_train=df_data.drop(y_name,axis=1)
            y_train=df_data[[y_name]]

        # Hiperparámetros del regresor
        param_grid = {'n_estimators': [100, 300, 500],'max_depth': [3, 5, 10],'learning_rate': [0.01, 0.1]}
        
        #define model
        regressor= XGBRegressor(random_state=self.seed) #jobs=-1
        
        #train model- Hiper parameter tunning

        tscv = TimeSeriesSplit(n_splits=n_folds,test_size=steps).split(X_train) #time series split
        grid_search = GridSearchCV(regressor, param_grid, scoring=['neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error'],cv=tscv, refit=metric_param,n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        #train model
        winner_model = grid_search.best_estimator_
 
        return winner_model, grid_search

    def predict_steps(self,model,X):
        #scale features 
        if self.scaler_X!= None: 
            X_scale=pd.DataFrame(self.scaler_X.transform(X),columns=self.scaler_X.feature_names_in_,index=X.index)
            predictions = model.predict(X_scale)
            pred_df = pd.DataFrame(self.scaler_y.inverse_transform(predictions.reshape(len(predictions),1)),columns=self.scaler_y.feature_names_in_,index=X.index)
        else:
            predictions = model.predict(X)
            pred_df = pd.DataFrame(predictions,columns=['twobe'],index=X.index)

        return pred_df