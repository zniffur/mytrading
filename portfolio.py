import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

WORKDIR = "csv/"
EXT = '.csv'
VOL_AVG_PERIOD = 20
etf_info_file = 'etf_info_vol' + str(VOL_AVG_PERIOD) + '_clean.csv'


def read_etf_info():
    df = pd.read_csv(etf_info_file)
    df['endDate'] = pd.to_datetime(df['endDate'])
    df['startDate'] = pd.to_datetime(df['startDate'])
    df.drop(df.columns[0], axis=1, inplace=True)

    return df


def read_quotes(lista):
    
    df2 = read_etf_info()
    ticker10 = [x.upper()+'.MI' for x in lista]
    df4 = df2[df2.ticker.isin(ticker10)]
    sources1 = df4.dataSource.tolist()

    ticker1 = df4.ticker.tolist()
    ticker1 = [x.lower().split('.')[0] for x in ticker1]

    df100 = pd.DataFrame()
    for source in sources1:
        filename = WORKDIR + source + EXT
        df_t = pd.read_csv(filename, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
        df100 = df100.join(df_t.close,how='outer',rsuffix='_'+source)
        
    df100.columns = ticker1
    datebuone = pd.DataFrame(columns=['primaData'])
    for col_name, data in df100.items(): 
        a = data.first_valid_index()
        datebuone.loc[col_name] = a

    prima_data_buona = datebuone.primaData.max()
    df101 = df100[df100.index >= prima_data_buona]

    # back-fill i NaN
    df102 = df101.fillna(method='bfill')

    return df102
