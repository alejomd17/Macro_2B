import os
import pandas as pd
import numpy as np
from datetime import datetime

print(os.path.abspath(os.path.join(os.path.dirname('__file__'))))
# os.chdir("..")
ROOT_PATH =  os.path.abspath(os.path.join(os.path.dirname('__file__')))
# Read config file
DATA_IN_PATH= os.path.join(ROOT_PATH,'data','input')
DATA_OUT_PATH= os.path.join(ROOT_PATH,'data','output')

def df_to_csv(df_data, filename):
    df_data.to_csv(os.path.join(DATA_OUT_PATH,f'{filename}.csv'))
    return df_data

##load data
def data_preprocessing(start_date, end_date):
    # 2B
    twobe = pd.read_excel(os.path.join(DATA_IN_PATH,'ventas_cantidad_canal_amz_2b.xlsx'))
    twobe = twobe[['Fecha','cantidad']]
    twobe.columns = ['Fecha', 'value']
    twobe["año"] = twobe["Fecha"].dt.year
    twobe["mes"] = twobe["Fecha"].dt.month
    twobe["fecha"] = twobe.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    twobe = pd.DataFrame(twobe.groupby('fecha')['value'].mean()).reset_index()
    twobe = twobe[['fecha','value']]
    twobe.columns = ['fecha', 'twobe']

    # Amazon
    amazon = pd.read_excel(os.path.join(DATA_IN_PATH,'amazon.xlsx'))
    amazon = amazon[['Fecha','Close']]
    amazon.columns = ['Fecha', 'value']
    amazon["año"] = amazon["Fecha"].dt.year
    amazon["mes"] = amazon["Fecha"].dt.month
    amazon["fecha"] = amazon.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    amazon = pd.DataFrame(amazon.groupby('fecha')['value'].mean()).reset_index()
    amazon = amazon[['fecha','value']]
    amazon.columns = ['fecha', 'amz']

    # Unemployment
    unemploy = pd.read_excel(os.path.join(DATA_IN_PATH,'unemployment.xlsx'))
    unemploy = unemploy[['Fecha','Actual']]
    unemploy.columns = ['Fecha', 'value']
    unemploy["año"] = unemploy["Fecha"].dt.year
    unemploy["mes"] = unemploy["Fecha"].dt.month
    unemploy["fecha"] = unemploy.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    unemploy = pd.DataFrame(unemploy.groupby('fecha')['value'].mean()).reset_index()
    unemploy = unemploy[['fecha','value']]
    unemploy.columns = ['fecha', 'unempl']
    
    # Federal Funds
    federal_funds = pd.read_excel(os.path.join(DATA_IN_PATH,'federal_founds.xlsx'))
    federal_funds = federal_funds[['Fecha','Último']]
    federal_funds.columns = ['Fecha', 'value']
    federal_funds["value"] = federal_funds["value"]/100
    federal_funds["año"] = federal_funds["Fecha"].dt.year
    federal_funds["mes"] = federal_funds["Fecha"].dt.month
    federal_funds["fecha"] = federal_funds.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    federal_funds = pd.DataFrame(federal_funds.groupby('fecha')['value'].mean()).reset_index()
    federal_funds = federal_funds[['fecha','value']]
    federal_funds.columns = ['fecha', 'federal_funds']

    # Apparel Retailers Downjones Index
    apparel_retailers_dj_index = pd.read_excel(os.path.join(DATA_IN_PATH,'apparel_retailers_downjones.xlsx'))
    apparel_retailers_dj_index = apparel_retailers_dj_index[['Fecha','Price']]
    apparel_retailers_dj_index.columns = ['Fecha', 'value']
    apparel_retailers_dj_index.value = apparel_retailers_dj_index.value.apply(int)
    apparel_retailers_dj_index["año"] = apparel_retailers_dj_index["Fecha"].dt.year
    apparel_retailers_dj_index["mes"] = apparel_retailers_dj_index["Fecha"].dt.month
    apparel_retailers_dj_index["fecha"] = apparel_retailers_dj_index.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    apparel_retailers_dj_index = pd.DataFrame(apparel_retailers_dj_index.groupby('fecha')['value'].mean()).reset_index()
    apparel_retailers_dj_index = apparel_retailers_dj_index[['fecha','value']]
    apparel_retailers_dj_index.columns = ['fecha', 'apparel_retailers']

    # Bond Yield 10 years
    tes_10y = pd.read_excel(os.path.join(DATA_IN_PATH,'bonds_10y.xlsx'))
    tes_10y = tes_10y[['Fecha','Último']]
    tes_10y.columns = ['Fecha', 'value']
    tes_10y["value"] = tes_10y["value"]/100
    tes_10y["año"] = tes_10y["Fecha"].dt.year
    tes_10y["mes"] = tes_10y["Fecha"].dt.month
    tes_10y["fecha"] = tes_10y.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    tes_10y = pd.DataFrame(tes_10y.groupby('fecha')['value'].mean()).reset_index()
    tes_10y = tes_10y[['fecha','value']]
    tes_10y.columns = ['fecha', 'test_10y']

    # GDP
    gdp = pd.read_excel(os.path.join(DATA_IN_PATH,'gdp.xlsx'))
    gdp = gdp[['Fecha','Actual']]
    gdp.columns = ['Fecha', 'value']
    gdp["año"] = gdp["Fecha"].dt.year
    gdp["mes"] = gdp["Fecha"].dt.month
    gdp["fecha"] = gdp.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    gdp = gdp[['fecha','value']]
    gdp.columns = ['fecha', 'gdp']

    # Inflation
    inflation = pd.read_excel(os.path.join(DATA_IN_PATH,'inflation_us.xlsx'))
    inflation = inflation[['Fecha','Actual']]
    inflation.columns = ['Fecha', 'value']
    inflation["año"] = inflation["Fecha"].dt.year
    inflation["mes"] = inflation["Fecha"].dt.month
    inflation["fecha"] = inflation.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    inflation = inflation[['fecha','value']]
    inflation.columns = ['fecha', 'inflation']

    # Consumer Discretionary SP500 Index
    consumer_discretionary_sp_index = pd.read_excel(os.path.join(DATA_IN_PATH,'consumer_discretionary_index_sp500.xlsx'))
    consumer_discretionary_sp_index = consumer_discretionary_sp_index[['Fecha','S&P 500 Consumer Discretionary (Sector)']]
    consumer_discretionary_sp_index.columns = ['Fecha', 'value']
    consumer_discretionary_sp_index["año"] = consumer_discretionary_sp_index["Fecha"].dt.year
    consumer_discretionary_sp_index["mes"] = consumer_discretionary_sp_index["Fecha"].dt.month
    consumer_discretionary_sp_index["fecha"] = consumer_discretionary_sp_index.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    consumer_discretionary_sp_index = pd.DataFrame(consumer_discretionary_sp_index.groupby('fecha')['value'].mean()).reset_index()
    consumer_discretionary_sp_index = consumer_discretionary_sp_index[['fecha','value']]
    consumer_discretionary_sp_index.columns = ['fecha', 'consumer_discretionary']

    # Global Luxury SP500 Index
    luxury_sp_index =pd.read_excel(os.path.join(DATA_IN_PATH,'global_luxury_index_sp500.xlsx'))
    luxury_sp_index = luxury_sp_index[['Fecha','S&P Global Luxury Index']]
    luxury_sp_index.columns = ['Fecha', 'value']
    luxury_sp_index["año"] = luxury_sp_index["Fecha"].dt.year
    luxury_sp_index["mes"] = luxury_sp_index["Fecha"].dt.month
    luxury_sp_index["fecha"] = luxury_sp_index.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    luxury_sp_index = pd.DataFrame(luxury_sp_index.groupby('fecha')['value'].mean()).reset_index()
    luxury_sp_index = luxury_sp_index[['fecha','value']]
    luxury_sp_index.columns = ['fecha', 'luxury']

    # FED Decisions
    decisiones_fed = pd.read_excel(os.path.join(DATA_IN_PATH,'decisiones_fed.xlsx'))
    decisiones_fed = decisiones_fed[['Fecha','Actual']]
    decisiones_fed.columns = ['Fecha', 'value']
    decisiones_fed["año"] = decisiones_fed["Fecha"].dt.year
    decisiones_fed["mes"] = decisiones_fed["Fecha"].dt.month
    decisiones_fed["fecha"] = decisiones_fed.apply(lambda row: str(row["año"])+"-" "0"+str(row["mes"]) \
                            if row["mes"]<10 else str(row["año"])+"-"+str(row["mes"]),axis=1)
    decisiones_fed = pd.DataFrame(decisiones_fed.groupby('fecha')['value'].mean()).reset_index()
    decisiones_fed = decisiones_fed[['fecha','value']]
    decisiones_fed.columns = ['fecha', 'fed_mp']
    
    list_df = [twobe, amazon, unemploy, federal_funds, apparel_retailers_dj_index, tes_10y,
                gdp,inflation,consumer_discretionary_sp_index,luxury_sp_index,decisiones_fed]

    df = list_df[0]
    for df_ in list_df[1:]:
        df = df.merge(df_, on='fecha', how = 'left')
    # df = df[(df.fecha >= '2015-01') & (df.fecha <= '2022-06')]
    df = df[(df.fecha >= start_date) & (df.fecha <= end_date)]
    df.fecha = df.fecha + '-01'
    df.fecha = pd.to_datetime(df.fecha, format="%Y/%m/%d")
    df_final = df.set_index('fecha')
    
    df_to_csv(df_final, 'macro_consolidate')
    return df_final

def main():
    start_date = '2015-01'
    end_date = '2022-06'
    pass