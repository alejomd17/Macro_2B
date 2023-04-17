# Libraries
# ==============================================================================
import pandas as pd
import numpy as np
from datetime import datetime
import os
from joblib import dump, load
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#models and aux functions

from .xgboost_regressor import XgboostRegressorModel
from .nn_regresor import MlpRegressorModel
from .lasso import LassoModel
from .auxiliar_functions import evaluate_metrics,select_best_model,obtain_steps,series_to_supervised

#Paths
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname('__file__')))
DATA_OUT_PATH = os.path.join(ROOT_PATH,'data','output')

def train_and_evaluate_models(data_in, features, div, model_list, lags_param, scaler_list, training_metric_list,steps_param,n_folds_param):
    """
    Funtion which allows train a model list iterate by a scales values list 
    inputs: 
        data : DataFrame-> dataset filtred by account
        steps : int -> steps to split data 
        models : list -> models object list
        scales : list -> scales columns names list in data_filtered 
    outputs: 
        trained_models : list-> tranined and evaluated models object list 
    """
    #data_train=data['train']
    #data_test=data['test']
    #df_all_data=data["data"]
    #features=data["features"]
    #div=len(data_test)
    
    #inicializar variables y objetos

    model_dic={}
    columns=['model_name','scaler','eval_training_metric','lags','k_fold','steps_param_trainig', 'best_params','eval_best_score', 'test_rmse','test_mape','test_r2']
    metric = training_metric_list[0]
    trained_evaluated_models_obj = []

    for model in model_list:
        df_results_grid=pd.DataFrame()
        count_model=0
        winner_name=[]
        model_dic={}
        if isinstance(model, MlpRegressorModel):
            scaler_list=[x for x in scaler_list if x !=None]
        for lags in range(1,lags_param+1):

            #create dataset with lags
            df_lags = series_to_supervised(data_in[features], n_lags=lags, n_out=1, dropnan=True)
            df=df_lags[df_lags.columns[:-len(features)+1]]
            df['mes']=df.index.month
            
            #Split dataFrame
            data=model.split_data(df,list(df.columns),div)

            #data_test=df[-div:]

            data_train = data['train']
            data_test = data['test']
            

            for scaler_param in scaler_list:

                for test_train_steps in range(1,steps_param+1):

                    winner, result_grid = model.train_model(data_train, steps=test_train_steps, y_name='twobe', metric_param=metric,scaler=scaler_param, n_folds=n_folds_param)
                    
                    #evaluate modelos with test data
                    predictions=model.predict_steps(winner,data_test.drop('twobe',axis=1))
                    
                    eval_metrics = evaluate_metrics(data_test[['twobe']], predictions)
                    
                    array=np.array([model.name+f"_{count_model}",str(scaler_param() if scaler_param!=None else None), metric,lags,n_folds_param,test_train_steps,result_grid.best_params_,result_grid.best_score_, eval_metrics['rmse'],eval_metrics['mape'],eval_metrics['r2'] ]).reshape(1,len(columns)) 
                    df=pd.DataFrame(array,columns=columns)


                    df_results_grid = pd.concat([df_results_grid,df])
                    df_results_grid.reset_index(inplace=True, drop=True)
                    
                    model_dic[f"{model.name}_{count_model}"]= [winner,model.scaler_X, model.scaler_y ,eval_metrics,lags,test_train_steps ]
                    count_model+=1

        df_results_grid['metrica_ponderada']=(df_results_grid['test_rmse']+df_results_grid['test_mape']-df_results_grid['test_r2'])/3
        df_results_grid.to_csv(os.path.join(DATA_OUT_PATH,'training_models',f'results_grid_{model.name}_{datetime.now().date().isoformat()}.csv'),sep=';', decimal=",")
        winner_name.append(df_results_grid['model_name'][df_results_grid['metrica_ponderada']==df_results_grid['metrica_ponderada'].min()].iloc[:1].values[0])
        winner_name.append(df_results_grid['model_name'][df_results_grid['test_rmse']==df_results_grid['test_rmse'].min()].iloc[:1].values[0])
        winner_name.append(df_results_grid['model_name'][df_results_grid['test_mape']==df_results_grid['test_mape'].min()].iloc[:1].values[0])
        winner_name.append(df_results_grid['model_name'][df_results_grid['test_r2']==df_results_grid['test_r2'].max()].iloc[:1].values[0])
        # Save winner models
        for w in winner_name:
            dump( model_dic[w][0], filename=os.path.join(DATA_OUT_PATH,'training_models',f'forecaster_{w}_{datetime.now().date().isoformat()}.py'))
            #dump( model_dic[w][1], filename=os.path.join(DATA_OUT_PATH,'training_models',f'model_class_{w}_{datetime.now().date().isoformat()}.py'))
            dump( model_dic[w][1], filename=os.path.join(DATA_OUT_PATH,'training_models',f'scaler_X_{w}_{datetime.now().date().isoformat()}.py'))
            dump( model_dic[w][2], filename=os.path.join(DATA_OUT_PATH,'training_models',f'scaler_y_{w}_{datetime.now().date().isoformat()}.py'))
            temp_dict={"model_name":w,"model":model_dic[w][0],"metrics":model_dic[w][3],"lags":model_dic[w][4],"test_train_steps":model_dic[w][5]}
            trained_evaluated_models_obj.append(temp_dict)

    return trained_evaluated_models_obj

def main(data, features, div, lags_param):

    
    # Train the models and Evaluate the metrics for each model
    #parameters
    model_list= [LassoModel(),XgboostRegressorModel(),MlpRegressorModel()]
    scaler_list=[None,MinMaxScaler,StandardScaler]
    training_metric_list=['neg_root_mean_squared_error','neg_mean_absolute_error','neg_mean_absolute_percentage_error']
    n_folds_param=3
    steps_param=1
    
    trained_models_list = train_and_evaluate_models(data, features, div, model_list, lags_param, scaler_list, training_metric_list, steps_param, n_folds_param)

    # Select the best model base on the metrics 
    
    winner = select_best_model(trained_models_list)

    #Save model
    #model_name=f"itr_model-{}-{}"

    return winner
