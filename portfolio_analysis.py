
#%%
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')


#%%
import sys
get_ipython().system('{sys.executable} -m pip install jinja2')

# QUESTO INSTALL xlrd, openpyxl all'interno del virtualenv
# FONDAMENTALE!!!!
#

get_ipython().system('{sys.executable} -m pip install xlrd')
get_ipython().system('{sys.executable} -m pip install openpyxl')
get_ipython().system('{sys.executable} -m pip install sklearn')
get_ipython().system('{sys.executable} -m pip install seaborn')



#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
import seaborn as sns; sns.reset_orig()

plt.style.use('ggplot')

WORKDIR = "csv/"
EXT = '.csv'
VOL_AVG_PERIOD = 20


#%%
print(plt.style.available)

#%% [markdown]
# ### Importo la lista degli ETF e ETC borsaitalia

#%%
dfEtfInfo = pd.read_excel('infoproviders.xlsx').iloc[6:,:]
dfEtfInfo.columns = dfEtfInfo.iloc[0,:].tolist()
dfEtfInfo = dfEtfInfo.iloc[1:,:]
dfEtfInfo.set_index('N.',inplace=True)

# df2['Strumento'].unique()
# df2.drop(df2[df2['Strumento'] == 0].index, inplace=True) # pulisce
dfEtfInfo = dfEtfInfo[dfEtfInfo['Strumento'] == 'ETF']
dfEtfInfo = dfEtfInfo[['ISIN','Nome','Local Market TIDM','Indice Benchmark','TER','Area Benchmark','Emittente','Strumento']]
dfEtfInfo.columns = ['isin','nome','ticker','benchmark','ter','areaBenchmark','emittente','strumento']
# df2.set_index('ticker', inplace=True)


#%%
dfEtcInfo = pd.read_excel('infoprovider_etc.xlsx').iloc[6:,:]
dfEtcInfo.columns = dfEtcInfo.iloc[0,:].tolist()
dfEtcInfo = dfEtcInfo.iloc[1:,:]
dfEtcInfo.set_index('N.',inplace=True)
dfEtcInfo = dfEtcInfo[['ISIN','Nome','Local Market TIDM','Indice Benchmark','TER','Area Benchmark','Emittente','Strumento']]
dfEtcInfo.columns = ['isin','nome','ticker','benchmark','ter','areaBenchmark','emittente','strumento']


#%%
df = pd.DataFrame()
df = pd.concat([dfEtfInfo, dfEtcInfo], ignore_index=True)
df.ticker = df.ticker + '.MI'


#%%
# capisce se è da usare il file ticker o l'isin come sorgente dei dati

def set_data_source(isin, ticker):
    isin_file = WORKDIR + isin + EXT
    if os.path.exists(isin_file):
        return isin
    else:
        ticker_file = WORKDIR + ticker + EXT
        if os.path.exists(ticker_file):
            return ticker
        else:
            return None
    
df['dataSource'] = df.apply(lambda x: set_data_source(x['isin'],x['ticker']), axis=1)    


#%%
df.dropna(inplace=True)  # droppa i ticker per cui non c'è nè il ISIN.csv, nè il TICKER.csv


#%%
## SALVO i df

df.to_excel('etf_info.xlsx')
df.to_csv('etf_info.csv')

#%% [markdown]
# ### Aggiungo 3 colonne con le date di inizio e fine delle quotazioni  e volAvg, disponibili sui file csv

#%%
df1 = df.copy()


#%%
def get_dates_volume(row):
    
    source = row['dataSource']  # isin o ticker
    try:
        ticker_file = WORKDIR + source + EXT
        df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
        end = df.index[df.shape[0]-1]
        start = df.index[0]
        volAvg = df.volume.rolling(VOL_AVG_PERIOD).mean()[-1]
    except e:
        end = None
        start = None
        volAvg = None

    return pd.Series([start,end,volAvg])

## LONG!!!
df1[['startDate', 'endDate', 'volAvg']] = df1.apply(get_dates_volume, axis=1)


#%%
df1['endDate'] =  pd.to_datetime(df1['endDate'])
df1['startDate'] =  pd.to_datetime(df1['startDate'])
df1 = df1.round({'volAvg':0})


#%%
## SALVO i df con volumi e ticker puliti

df1.to_excel('etf_info_vol' + str(VOL_AVG_PERIOD)  + '.xlsx')
df1.to_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '.csv')


#%% [markdown]
# ### elimina i ticker con end_date vecchia e quelli con volume NaN (che vuol dire che non esistono file per quel ticker)

#%%
df2 = df1.copy()

mask = (df2.endDate < '2019')
df2.drop(df2[mask].index, inplace=True)

df2 = df2[df2.volAvg >= 0]

df2.volAvg = df2.volAvg.astype(int)
df2.ter = df2.ter.astype(float)


#%%
## SALVO i df con volumi e ticker puliti

df2.to_excel('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean.xlsx')
df2.to_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean.csv')

#%% [markdown]
# ## CARICA il DF (reinizializza)

#%%
df = pd.read_csv('etf_info' + '.csv')
df.drop(df.columns[0] ,axis=1, inplace=True)
df.head()


#%%
df1 = pd.read_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '.csv')
df1['endDate'] =  pd.to_datetime(df1['endDate'])
df1['startDate'] =  pd.to_datetime(df1['startDate'])
df1.drop(df1.columns[0] ,axis=1, inplace=True)
df1.head()


#%%
df2 = pd.read_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean.csv')
df2['endDate'] =  pd.to_datetime(df2['endDate'])
df2['startDate'] =  pd.to_datetime(df2['startDate'])
df2.drop(df2.columns[0] ,axis=1, inplace=True)
df2.head()


#%%
df3 = pd.read_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean_mach1.csv')
df3['endDate'] =  pd.to_datetime(df3['endDate'])
df3['startDate'] =  pd.to_datetime(df3['startDate'])
df3.drop(df3.columns[0] ,axis=1, inplace=True)
df3.head()


#%%
df102 = pd.read_csv('quotes_clean_mach1.csv')
df102['dateTime'] = pd.to_datetime(df102['dateTime'])
df102.set_index('dateTime', inplace=True)
df102.head()

#%% [markdown]
# # Analisi dati ETF

#%%
df2.describe()


#%%
df2.volAvg.quantile(q=0.85)


#%%
# salva i nomi delle aree benchmark in un file
with open('area.csv', "w") as outfile:
    for entries in df.areaBenchmark.unique():
        outfile.write(entries)
        outfile.write("\n")


#%%
# costruita a mano dal file area.csv

listaAree = ['AZIONARIO EUROPA - AREA',
'AZIONARIO MONDO',
'AZIONARIO NORD AMERICA',
'COMMODITIES',
'METALLI PREZIOSI',
'REAL ESTATE',
'TITOLI DI STATO - EURO',
'TITOLI DI STATO - MONDO',
'TITOLI DI STATO - NON EURO']


#%%
df3 = df2[df2.areaBenchmark.isin(listaAree)]


#%%
# salva il df3, con solo i ticker interessanti

df3.to_excel('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean_mach1.xlsx')
df3.to_csv('etf_info_vol' + str(VOL_AVG_PERIOD)  + '_clean_mach1.csv')

#%% [markdown]
# ### LISTA 1 - az US, az EU, obb EU, obb US, gold

#%%
# lista 1 ticker 'buoni', costruita a mano partendo da mach1

lista1 = ['imeu','ceu','smea','iusa','csspx','phau','ibtm','ibgm', 'emg', 'em15', 'ibgl']
ticker10 = [x.upper()+'.MI' for x in lista1]


#%%
df4 = df2[df2.ticker.isin(ticker10)]

#%% [markdown]
# #### Dataframe con le info dei ticker "buoni"

#%%
df4

#%% [markdown]
# #### Creo un DF con le quotazioni close dei ticker selezionati

#%%
sources1 = df4.dataSource.tolist()

ticker1 = df4.ticker.tolist()
ticker1 = [x.lower().split('.')[0] for x in ticker1]
ticker1


#%%
dfLista = pd.DataFrame()
for source in sources1:
    filename = WORKDIR + source + EXT
    df_t = pd.read_csv(filename, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
    dfLista = dfLista.join(df_t.close,how='outer',rsuffix='_'+source)
    
dfLista.columns = ticker1


#%%
df100 = dfLista.copy()

#%% [markdown]
# #### Correzione SMEA

#%%
df100.smea.idxmin()


#%%
df100.smea.loc['2018-11-22']=df100.smea.loc['2018-11-21']

#%% [markdown]
# ### Pulizia dati (NaN)

#%%
for col_name, data in df100.items(): 
    print("First valid index for column {} is at {}".format(col_name, data.first_valid_index()))


#%%
# prima data valida 
df101 = df100[df100.index >= '2010-05-26']

# back-fill i NaN
df102 = df101.fillna(method='bfill')


#%%
# SALVO df con quotazioni titoli selezionati puliti

df102.to_csv('quotes_clean_mach1.csv')
df102.to_excel('quotes_clean_mach1.xlsx')

#%% [markdown]
# ### Sampling a 'week' (df200)

#%%
df103 = df102.copy()
df103['weekday'] = df103.index.weekday_name


#%%
# l'offset serve per allineare la data ai venerdì in cui prende il campione
df200 = df103.resample('W',loffset=pd.offsets.timedelta(days=-2)).last()

# droppo la colonna weekday per normalizzare
df200 = df200.drop(columns='weekday')

# normalizzo e plotto

norm_df200 = df200/df200.iloc[0]
#norm_df200.plot()


#%%
plt.style.use('default')
#norm_df200.plot(kind='line', colormap='PiYG')
#norm_df200.plot(kind='line', colormap='inferno')
norm_df200.plot(kind='line', colormap='Paired', figsize=(9,7))

#%% [markdown]
# ### Returns

#%%
returns = np.log(df200/df200.shift(1))
returns.dropna(inplace=True)


#%%
PERIODS = 52
stats = pd.DataFrame()
stats['Annualized Returns(%)'] = returns.mean() * PERIODS *100
stats['Annualized Volatility(%)'] = returns.std() * np.sqrt(PERIODS)*100
stats['Sharpe Ratio'] = stats['Annualized Returns(%)'] /stats['Annualized Volatility(%)']
print(82*'-')
print('Assets Classes Annualized Statistics — full observation period')
stats.style.bar(color=['red','green'], align='zero')


#%%
returns.corr('pearson')


#%%
sns.set()

fig4 = plt.figure()
#sns.distplot(returns['ibtm'])
sns.distplot(returns['em15'])
sns.distplot(returns['ibgl'])
#plt.legend(('ibtm','em15','ibgm'),fontsize = 12)
plt.legend(('em15','ibgl'),fontsize = 12)
plt.show()

#%% [markdown]
# ## Portfolio allocation(s)
#%% [markdown]
# #### Preparo la matrice delle allocations

#%%
allocation = pd.DataFrame(index=norm_df200.columns)
allocation['ticker'] = allocation.index + '.MI'
allocation.ticker = allocation.ticker.str.upper()

df4ri = df4.set_index('ticker')
def myfunc(row):
    return df4ri.loc[row.ticker,'areaBenchmark']

allocation['area'] = allocation.apply(myfunc, axis=1)

allocation['one'] = 0
allocation['two'] = 0

allocation

#%% [markdown]
# #### setup allocazioni

#%%
# one
allocation.loc['iusa','one'] = 0.2
allocation.loc['smea','one'] = 0.2
allocation.loc['phau','one'] = 0.1
allocation.loc['em15','one'] = 0.2
allocation.loc['ibtm','one'] = 0.3
# two
allocation.loc['iusa','two'] = 0.3
allocation.loc['smea','two'] = 0.2
allocation.loc['phau','two'] = 0.1
allocation.loc['em15','two'] = 0.3
allocation.loc['ibtm','two'] = 0.1


#%%
#norm_df200.ceu.plot()
#norm_df200.smea.plot()
#norm_df200.imeu.plot()
#plt.legend()

#norm_df200.em15.plot()
#norm_df200.ibgl.plot()
#norm_df200.emg.plot()
#plt.legend()


#%%
lEtfs = allocation[allocation.one != 0].index.tolist()


#%%
allocation_restr = allocation[allocation.one != 0]
allocation_restr

#%% [markdown]
# ### Calcolo rendimento di due portafogli "miei" - no rebalancing

#%%
norm_df201 = norm_df200.copy()


#%%
norm_df201['one'] = norm_df200.mul(allocation.one.values,axis=1).sum(axis=1)
norm_df201['two'] = norm_df200.mul(allocation.two.values,axis=1).sum(axis=1)


#%%
norm_df201.one.plot()
norm_df201.two.plot()
norm_df201.csspx.plot()  # per confronto, plotto il sp500
plt.legend()


#%%
exp_ret_one = np.sum(returns.mean()* allocation.one)* PERIODS
exp_std_one = np.sqrt(np.dot(allocation.one.T,np.dot(returns.cov()*PERIODS, allocation.one)))
sharpe_one = exp_ret_one/exp_std_one

exp_ret_two = np.sum(returns.mean()* allocation.two)* PERIODS
exp_std_two = np.sqrt(np.dot(allocation.two.T,np.dot(returns.cov()*PERIODS, allocation.two)))
sharpe_two = exp_ret_two/exp_std_two

print('Key Stats: Portfolio one')
print(82*'=')
print('Annualized Returns: {:.3%}'.format(exp_ret_one))
print('Annualized Volatility: {:.3%}'.format(exp_std_one))
print('Sharpe Ratio: {:.4}'.format(sharpe_one))
print(82*'-')
print('Key Stats: Portfolio two ')
print(82*'=')
print('Annualized Returns: {:.3%}'.format(exp_ret_two))
print('Annualized Volatility: {:.3%}'.format(exp_std_two))
print('Sharpe Ratio: {:.4}'.format(sharpe_two))
print(82*'-')


#%%
binsnumber = 35
fig7, ax = plt.subplots(figsize=(14,10))
plt.subplots_adjust(hspace=.4,wspace=.4) # it adds space in between plots
plt.subplot(121)
ax = plt.gca()

ax.hist(norm_df201['one'], bins=binsnumber, color='steelblue', density = True,
       alpha = 0.5, histtype ='stepfilled',edgecolor ='red' )

sigma, mu = norm_df201['one'].std(),norm_df201['one'].mean() # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, binsnumber, density=True, alpha = 0.1)
ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=1.5, color='r')

ax.annotate('Skewness: {}\n\nKurtosis: {}'.format(round(norm_df201['one'].skew(),2),round(norm_df201['one'].kurtosis(),2)),
             xy=(10,20),xycoords = 'axes points',xytext =(20,360),fontsize=14)


ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.set_title('Portfolio one')

plt.subplot(122)
ax1 = plt.gca()

ax1.hist(norm_df201['two'], bins=binsnumber, color='steelblue', density = True,
       alpha = 0.5, histtype ='stepfilled',edgecolor ='red' )

sigma, mu = norm_df201['two'].std(),norm_df201['two'].mean() # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, binsnumber, density=True, alpha = 0.1)
ax1.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=1.5, color='r')

ax1.annotate('Skewness: {}\n\nKurtosis: {}'.format(round(norm_df201['two'].skew(),2),round(norm_df201['two'].kurtosis(),2)),
             xy=(10,20),xycoords = 'axes points',xytext =(20,360),fontsize=14)


ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Portfolio two')

plt.show();

#%% [markdown]
# ## Montecarlo

#%%
allocation_restr


#%%
myrets = np.array([exp_ret_one, exp_ret_two])
myvols = np.array([exp_std_one, exp_std_two])


#%%
lista = lEtfs.copy()
lista.append('returns')
lista.append('volatility')
lista.append('sharpe')

monte_df = pd.DataFrame(columns=lista)
for p in range(2500):
    weights = np.random.random(len(allocation_restr))
    weights /= np.sum(weights)

    ret = np.sum(returns[lEtfs].mean()*PERIODS*weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(returns[lEtfs].cov()*PERIODS, weights)))
    #monte_df.loc[p] = pandas.Series({'a':1, 'b':5, 'c':2, 'd':3})
    sharpe = ret / vol

    ww = weights.tolist()
    ww.append(ret)
    ww.append(vol)
    ww.append(sharpe)

    monte_df.loc[p] = ww


#%%
fig9 = plt.figure(figsize = (8,6))

plt.scatter(monte_df.volatility, monte_df.returns, 
            c = monte_df.sharpe, 
            marker = 'o', cmap='coolwarm')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label = 'Sharpe Ratio')
plt.title('Monte Carlo Simulation Efficient Frontier')

plt.scatter(myvols, myrets, c = myrets / myvols, marker = '+', cmap='coolwarm')

plt.show();


#%%
monte_df.head()


#%%
monte_df.iloc[monte_df.sharpe.idxmax()]


#%%
monte_df.iloc[monte_df.volatility.idxmin()]

#%% [markdown]
# ## Montecarlo 2 - calcolo efficient frontier

#%%
assets_all = ['imeu','ceu','smea','iusa','csspx','phau','ibtm','ibgm', 'emg', 'em15', 'ibgl']
assets = ['em15', 'ibtm', 'iusa', 'smea', 'phau']
numAssets = len(assets)
PERIODS = 252
dur = 20
riskFreeRate = 0.01


#%%
data = df102.copy()  # daily quotes assets_all

## LOG *DAILY* returns
returns = np.log(data/data.shift(1))
returns.dropna(inplace=True)

meanDailyReturns = returns[assets].mean()
covMatrix = returns[assets].cov()
meanDailyReturns


#%%
def calcPortfolioPerf(weights, meanReturns, covMatrix, periods):
    '''
    Calculates the expected mean of returns and volatility for a portolio of
    assets, each carrying the weight specified by weights

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    tuple containing the portfolio return and volatility
    '''    
    #Calculate return and variance

    portReturn = np.sum( meanReturns*weights*periods )
    portStdDev = np.sqrt(np.dot(weights.T, np.dot(covMatrix*periods, weights)))

    return portReturn, portStdDev


#%%

## TEST su 2 portafogli disegnati da me

# assets = ['em15', 'ibtm', 'iusa', 'smea', 'phau']
weights = np.array([0.2,0.3,0.2,0.2,0.1])
weights = np.array([0.3,0.1,0.3,0.2,0.1])

calcPortfolioPerf(weights, meanDailyReturns, covMatrix, PERIODS)


#%%
#Run MC simulation of numPortfolios portfolios

numPortfolios = 2500
results = np.zeros((3,numPortfolios))

#Calculate portfolios

for i in range(numPortfolios):
    #Draw numAssets random numbers and normalize them to be the portfolio weights

    weights = np.random.random(numAssets)
    weights /= np.sum(weights)

    #Calculate expected return and volatility of portfolio

    pret, pvar = calcPortfolioPerf(weights, meanDailyReturns, covMatrix, PERIODS)

    #Convert results to annual basis, calculate Sharpe Ratio, and store them

    results[0,i] = pret
    results[1,i] = pvar
    results[2,i] = (results[0,i] - riskFreeRate)/results[1,i]


#%%
import scipy.optimize as sco

def negSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate, periods):
    '''
    Returns the negated Sharpe Ratio for the speicified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    p_ret, p_var = calcPortfolioPerf(weights, meanReturns, covMatrix, periods)

    return -(p_ret - riskFreeRate) / p_var

def getPortfolioVol(weights, meanReturns, covMatrix, periods):
    '''
    Returns the volatility of the specified portfolio of assets

    INPUT
    weights: array specifying the weight of each asset in the portfolio
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio

    OUTPUT
    The portfolio's volatility
    '''
    return calcPortfolioPerf(weights, meanReturns, covMatrix, periods)[1]

def findMaxSharpeRatioPortfolio(meanReturns, covMatrix, riskFreeRate, periods):
    '''
    Finds the portfolio of assets providing the maximum Sharpe Ratio

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    riskFreeRate: time value of money
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate, periods)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(negSharpeRatio, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts

def findMinVariancePortfolio(meanReturns, covMatrix, periods):
    '''
    Finds the portfolio of assets providing the lowest volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, periods)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(numAssets))

    opts = sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return opts


#%%
maxPtfls = pd.DataFrame(columns=['ret','vola'])

#Find portfolio with maximum Sharpe ratio
maxSharpe = findMaxSharpeRatioPortfolio( meanDailyReturns, covMatrix,
                                        riskFreeRate, PERIODS)
rp, sdp = calcPortfolioPerf(maxSharpe['x'], meanDailyReturns, covMatrix, PERIODS)
a0 = np.array([rp, sdp])
maxPtfls.loc[0] = a0

#Find portfolio with minimum variance
minVar = findMinVariancePortfolio(meanDailyReturns, covMatrix, PERIODS)
rp, sdp = calcPortfolioPerf(minVar['x'], meanDailyReturns, covMatrix, PERIODS)
a1 = np.array([rp, sdp])
maxPtfls.loc[1] = a1


#%%
def findEfficientReturn(meanReturns, covMatrix, targetReturn, periods):
    '''
    Finds the portfolio of assets providing the target return with lowest
    volatility

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, periods)

    def getPortfolioReturn(weights):
        return calcPortfolioPerf(weights, meanReturns, covMatrix, periods)[0]

    constraints = ({'type': 'eq', 'fun': lambda x: getPortfolioReturn(x) - targetReturn},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(numAssets))

    return sco.minimize(getPortfolioVol, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)

def findEfficientFrontier(meanReturns, covMatrix, rangeOfReturns, periods):
    '''
    Finds the set of portfolios comprising the efficient frontier

    INPUT
    meanReturns: mean values of each asset's returns
    covMatrix: covariance of each asset in the portfolio
    targetReturn: APR of target expected return

    OUTPUT
    Dictionary of results from optimization
    '''
    efficientPortfolios = []
    for ret in rangeOfReturns:
        efficientPortfolios.append(findEfficientReturn(meanReturns, covMatrix, ret, periods))

    return efficientPortfolios

#Find efficient frontier, annual target returns of 9% and 16% are converted to

#match period of mean returns calculated previously


#%%
#targetReturns = np.linspace(0.09, 0.16, 50)/(252./dur)
targetReturns = np.linspace(0.02, 0.12, 20)
efficientPortfolios = findEfficientFrontier(meanDailyReturns, covMatrix, targetReturns, PERIODS)


#%%
### PLOTTING

dfResults = pd.DataFrame(results.T)
dfResults.columns = ['ret','vola','sharpe']


#%%
fig10 = plt.figure(figsize = (8,6))


plt.scatter(dfResults.vola, dfResults.ret, c = dfResults.sharpe, 
            marker = 'o', cmap='plasma')

plt.scatter(maxPtfls.vola, maxPtfls.ret, c='red', marker = '+')

plt.plot([p['fun'] for p in efficientPortfolios], 
         targetReturns, marker='x')

plt.show();


#%%
#plt.figure(figsize=(8,6))
ind = np.arange(numAssets)
width = 0.1
fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.bar(ind, maxSharpe['x'], width, color='r', alpha=0.75)
rects2 = ax.bar(ind + width, minVar['x'], width, color='b', alpha=0.75)
rects3 = ax.bar(ind + 2*width,np.array([0.2,0.3,0.2,0.2,0.1]), width, color='g', alpha=0.75)
rects4 = ax.bar(ind + 3*width,np.array([0.3,0.1,0.3,0.2,0.1]), width, color='w', alpha=0.75)
rects5 = ax.bar(ind + 4*width,np.array([0.6,0.0,0.39,0.0,0.01]), width, color='black', alpha=0.75)

ax.set_ylabel('Weight of Asset in Portfolio')
ax.set_ylim(0,0.6)
ax.set_title('Comparison of Portfolio Compositions')
ax.set_xticks(ind + width)
ax.set_xticklabels(assets)
plt.tight_layout()
ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), 
          ('Max Sharpe Ratio', 'Minimum Volatility','one','two','three'))
#plt.savefig('Portfolio Compositions', dpi=100)
plt.show()


#%%
maxSharpe.x, minVar.x


#%%
# controprova
# assets = ['em15', 'ibtm', 'iusa', 'smea', 'phau']
weights = np.array([0.6,0.0,0.39,0.0,0.01])

calcPortfolioPerf(weights, meanDailyReturns, covMatrix, PERIODS)



#%%
## plot storico portafogli
# assets = ['em15', 'ibtm', 'iusa', 'smea', 'phau']
mydata = data[assets]
ndata = mydata/mydata.iloc[0]
ndata2 = ndata.copy()

weights = np.array([0.2,0.3,0.2,0.2,0.1])
ndata2['one'] = ndata.mul(weights,axis=1).sum(axis=1)

weights = np.array([0.3,0.1,0.3,0.2,0.1])
ndata2['two'] = ndata.mul(weights,axis=1).sum(axis=1)

weights = np.array([0.6,0.0,0.39,0.0,0.01])
ndata2['three'] = ndata.mul(weights,axis=1).sum(axis=1)

weights = minVar['x']
ndata2['minvar'] = ndata.mul(weights,axis=1).sum(axis=1)


#%%
# assets = ['em15', 'ibtm', 'iusa', 'smea', 'phau']
fig11 = plt.figure(figsize = (8,6))

plt.plot(ndata2.iusa)
plt.plot(ndata2.one)
plt.plot(ndata2.two)
plt.plot(ndata2.three)
plt.plot(ndata2.minvar)

plt.legend()
plt.show();


#%%
dd = ndata2.sub(ndata2.cummax())

dd.one.cummin().plot()
dd.two.cummin().plot()
dd.three.cummin().plot()
dd.minvar.cummin().plot()
dd.iusa.cummin().plot()
plt.legend()

#%% [markdown]
# # TESTs

#%%
datos_returns = np.log(datos/datos.shift(1))
datos_returns.dropna(inplace=True)
stats = pd.DataFrame()
stats[‘Annualized Returns(%)’] =datos_returns.mean() * semana *100
stats[‘Annualized Volatility(%)’] = datos_returns.std() * np.sqrt(semana)*100
stats[‘Sharpe Ratio’] = stats[‘Annualized Returns(%)’] /stats[‘Annualized Volatility(%)’]
print(82*’-’)
print(‘Assets Classes Annualized Statistics — full observation period’)
stats.style.bar(color=[‘red’,’green’], align=’zero’)


#%%



#%%
f = plt.figure()
plt.title('Title here!', color='black')
dfLista.ceu.plot(kind='line', ax=f.gca())
dfLista.imeu.plot(kind='line', ax=f.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()



#%%
z.loc[mask,'end_date'] = z[mask].apply(lambda x: correct_end_date(x['ticker'],x['isin']), axis=1)


#%%
z[mask]


#%%
def myfunc(ticker, isin):
    return str(ticker)+str(isin)

z.loc[mask,'end_date'] = z.apply(lambda x: myfunc(x['isin'],x['ticker']), axis=1)

#%% [markdown]
# ### Test lettura quotazioni ETF da ./csv/

#%%
tickers = ['A500.MI']


#%%
ticker_file = WORKDIR + tickers[0] + EXT

#df = pd.read_csv(ticker_file, index_col=1, parse_dates=True)
df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)


#%%
df.head()


#%%
end = df.index[df.shape[0]-1]
start = df.index[0]

end - start
end.date()

np.busday_count( start.date(), end.date() )

#%% [markdown]
# ### correzioni

#%%
tickers = ['EUE.MI']


#%%
ticker_file = WORKDIR + tickers[0] + EXT

#df = pd.read_csv(ticker_file, index_col=1, parse_dates=True)
df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)


#%%
df.loc['2018-11-21','high'] = 31.98
df.loc['2018-11-21','low'] = 31.69
df.loc['2018-11-21','close'] = 31.96



#%%
df.close.plot(figsize=(16, 10))

#%% [markdown]
# ### apply colonne multiple

#%%
df100 = dfEtfInfo


#%%
def myfunc(isin, ticker):
    return str(isin)+str(ticker)

df100['newcolumn'] = df100.apply(lambda x: myfunc(x['isin'],x['ticker']), axis=1)


#%%
df100.head()

#%% [markdown]
# ### leggi ticker.mi csv altrimenti isin.csv

#%%
def get_start_date(isin, ticker):
    try:
        ticker_file = WORKDIR + ticker + EXT
        df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True)
        end = df.index[df.shape[0]-1]
        start = df.index[0]
        return start
    except FileNotFoundError:
        try:
            ticker_file = WORKDIR + isin + EXT
            df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True)
            end = df.index[df.shape[0]-1]
            start = df.index[0]
            return start
        except FileNotFoundError:
            return None

def get_end_date(isin, ticker):
    try:
        ticker_file = WORKDIR + ticker + EXT
        df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True)
        end = df.index[df.shape[0]-1]
        start = df.index[0]
        return end
    except FileNotFoundError:
        try:
            ticker_file = WORKDIR + isin + EXT
            df = pd.read_csv(ticker_file, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True)
            end = df.index[df.shape[0]-1]
            start = df.index[0]
            return end
        except FileNotFoundError:
            return None


#%%
df100 = dfEtfInfo[dfEtfInfo.areaBenchmark == 'AZIONARIO NORD AMERICA']

df101 = df3[df3.benchmark == 'S&P 500 TRN USD']

df101['start_date'] = df101.apply(lambda x: get_start_date(x['isin'],x['ticker']), axis=1)
df101['end_date'] = df101.apply(lambda x: get_end_date(x['isin'],x['ticker']), axis=1)


#%%
df101.head()

#%% [markdown]
# ### Lista le aree degli ETF

#%%
aree = pd.DataFrame(dfEtfInfo.areaBenchmark.unique(),columns=['areaBenchmark'])
aree.sort_values(by='areaBenchmark',inplace=True)

aree


#%%
df3 = dfEtfInfo[dfEtfInfo.areaBenchmark == 'AZIONARIO NORD AMERICA']


#%%
df3[df3.benchmark == 'S&P 500 TRN USD']


#%%
df4 = df3[df3.benchmark == 'S&P 500 TRN USD']
# df5 = df4.assign(startDate=df4.ticker.apply(get_start_date),endDate=df4.ticker.apply(get_end_date))


#%%
df4['startDate'] = df4.ticker.apply(get_start_date)
df4['endDate'] = df4.ticker.apply(get_end_date)
    
    


#%%
def myfunc100(row):
    return pd.Series(['PIP','PO'])


dfEtfInfo[['test1', 'test2']] = dfEtfInfo.apply(myfunc100, axis=1)


#%%
dfEtfInfo = pd.read_excel('infoprovider_etc.xlsx').iloc[6:,:]
dfEtfInfo.columns = dfEtfInfo.iloc[0,:].tolist()
dfEtfInfo = dfEtfInfo.iloc[1:,:]
dfEtfInfo.set_index('N.',inplace=True)
# df2['Strumento'].unique()
# df2.drop(df2[df2['Strumento'] == 0].index, inplace=True) # pulisce
dfEtfInfo = dfEtfInfo[dfEtfInfo['Strumento'] == 'ETF']
dfEtfInfo = dfEtfInfo[['ISIN','Nome','Reuters RIC (Italy)','Indice Benchmark','TER','Area Benchmark','Emittente']]
dfEtfInfo.columns = ['isin','nome','ticker','benchmark','ter','areaBenchmark','emittente']
# df2.set_index('ticker', inplace=True)


#%%
dfEtcInfo


#%%
ceu = pd.read_csv('csv/CEU.MI.csv', usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
imeu = pd.read_csv('csv/IMEU.MI.csv', usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
tmp1 = ceu.join(imeu, how='outer', rsuffix='_1')[['close','close_1']]
tmp1.columns=['ceu','imeu']
tmp1


#%%
df5 = pd.DataFrame()

for index,row in df4.iterrows():
    ticker = row.ticker
    source = row.dataSource
    filename = WORKDIR + source + EXT
    tmpdf = pd.read_csv(filename, usecols=[1,2,3,4,5,6], index_col=0, parse_dates=True, dayfirst=True)
    df5[ticker] = tmpdf.close


