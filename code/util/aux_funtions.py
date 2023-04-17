import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def stat_significance_Correlation(df,features, p_value_limit=0.08):
    col =['corr_pearson', 'p_value_two-sided_p', 'p_value_greater_p', 'p_value_less_p', 'corr_spearman',
        'p_value_two-sided_s', 'p_value_greater_s', 'p_value_less_s', 'corr_kendalltau',
        'p_value_two-sided_k', 'p_value_greater_k', 'p_value_less_k']

    df_corr=pd.DataFrame([])
    for f in features:
        pearson=[]
        spearman=[]
        kendalltau=[]
        for h in ['two-sided', 'greater', 'less']:
            if h=='two-sided':
                r, p = stats.pearsonr(df['twobe'], df[f])
                pearson.append(r)
                pearson.append(p)
                r, p = stats.spearmanr(df['twobe'], df[f],alternative=h)
                spearman.append(r)
                spearman.append(p)
                r, p = stats.kendalltau(df['twobe'], df[f],alternative=h)
                kendalltau.append(r)
                kendalltau.append(p)
            else:
                r, p = stats.pearsonr(df['twobe'], df[f])
                pearson.append(p)
                r, p = stats.spearmanr(df['twobe'], df[f],alternative=h)
                spearman.append(p)
                r, p = stats.kendalltau(df['twobe'], df[f],alternative=h)
                kendalltau.append(p)
        
        df_corr_temp=pd.DataFrame([pearson+spearman+kendalltau], columns=col, index=[f])
        df_corr= pd.concat([df_corr,df_corr_temp])
    
    #correlaciones significativas p-value <0.05
    
    for c in ['corr_pearson','corr_spearman','corr_kendalltau']:
        print(df_corr[df_corr.apply(lambda x: x[f'p_value_greater_{c[5]}']<p_value_limit or x[f'p_value_less_{c[5]}']<p_value_limit or x[f'p_value_less_{c[5]}']<p_value_limit, axis=1)][c])


def lags_evaluation(df_data,y_list):
    features=list(df_data.columns[1:])
    pruebas = pd.DataFrame()
    for twobe in y_list:
        for macro in features:
            pruebas_temp = pd.DataFrame()
            justit = list(sm.tsa.stattools.ccf(df_data[twobe], df_data[macro], adjusted=False))[1:12]
            justit2=np.abs(justit)
            hoctus = list(justit2).index(max(justit2))
            pruebas_temp['feature'] = [macro]
            pruebas_temp['correlation_values'] = [np.around(justit , decimals=2) ]
            pruebas_temp['best_lag'] = [hoctus+1]
            pruebas_temp['value'] = [justit[hoctus]]
            pruebas = pd.concat([pruebas, pruebas_temp], axis = 0)
        pruebas.set_index('feature', inplace=True)
    return pruebas