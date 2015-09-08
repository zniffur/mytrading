'''
Created on 04/set/2015

@author: crisimo
'''

import get_etf_tickers 
import pandas as pd

import numpy as np
import datetime
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    tickers = ['AAPL', 'MSFT', 'MDY','IEV','EEM','ILF','EPP','EDV','SHY']
    tickers = ['MDY','IEV','EEM','ILF','EPP','EDV']
    it_more_vola_tickers = ['IUSA.MI', 'ETFMIB.MI', 'IJPN.MI', 'EUE.MI', 'MSE.MI',
                             'ISF.MI', 'DAXX.MI', 'IEEM.MI', 'IBZL.MI', 'IWRD.MI',
                              'EQQQ.MI', 'EUN.MI']
    
    '''
    # All ETFs Milano
    etfs, tickers = get_etf_tickers.get_etf_tickers()
    
    # Oldest ETFs Milano
    df = pd.read_csv('csv/oldest_etfs.csv')
    tickers = df['Reuters RIC (Italy)'].tolist()
    '''
    
    start = datetime.date(2003, 1, 1)
    end = datetime.date(2013, 07, 31)
      
    all_data = get_etf_tickers.get(tickers, start, end)
    
    '''
    # Loading data from files instead
    all_data = get_etf_tickers.load(tickers)
    '''
    
    # reset the index to make everything columns
    just_closing_prices = all_data[['Adj Close']].reset_index()
    
    # We moved the dates into a column because we now want to pivot Date into the 
    # index and each Ticker value into a column:
    daily_close_px = just_closing_prices.pivot('Date', 'Ticker', 'Adj Close')
    
    #daily_pct_change_2 = daily_close_px / daily_close_px.shift(1) - 1
    daily_pct_change = daily_close_px.pct_change()
    daily_pct_change.fillna(0, inplace=True)
    cum_daily_return = (1 + daily_pct_change).cumprod()
    
    '''
    cum_daily_return.plot(figsize=(12,8))
    plt.legend(loc=2);
    '''
    
    # ------------------ plot daily returns --------------------------
    # single stock histogram (distribution of return)
    aapl = daily_pct_change['AAPL']
    aapl.hist(bins=50, figsize=(12,8))
    aapl.describe(percentiles=[0.025, 0.5, 0.975])
    
    #comparative on all stocks
    daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
    daily_pct_change.plot(kind='box', figsize=(12,8));
    
    # scatterplot between two stocks
    limits = [-0.15, 0.15]
    get_etf_tickers.render_scatter_plot(daily_pct_change, 'MDY', 'AAPL', xlim=limits)
    
    # all stocks against each other, with a KDE in the diagonal
    _ = pd.scatter_matrix(daily_pct_change, diagonal='kde', alpha=0.1, figsize=(12,12));    
    
    
    # Q-Q plot vs normal distro
    import scipy.stats as stats
    f = plt.figure(figsize=(12,8))
    ax = f.add_subplot(111)
    stats.probplot(aapl, dist='norm', plot=ax)
    plt.show();
    
    # Volatility 
    min_periods = 75
    vol = pd.rolling_std(daily_pct_change, min_periods) * np.sqrt(min_periods)
    vol.plot(figsize=(10, 8))
	
    # Correlation - fixed, over the whole period
    daily_pct_change.corr()
    # - fixed, on a subset
    daily_pct_change['2012':'2013'].corr()
    # - rolling, on an year period, on 2 stocks
    rolling_corr = pd.rolling_corr(daily_pct_change['AAPL'], 
                                   daily_pct_change['MSFT'], window=252).dropna()
	
    # monthly calculations
    aaplM = aapl.resample('M', how='last')
    aaplMpct_chg = aaplM.pct_change()
    aaplMpct_chg.hist(bins=50, figsize=(12,8));
    aaplMpct_chg.describe(percentiles=[0.025, 0.5, 0.975])
    
    # ----- get some info on more volatile stocks -----
    volumes = all_data[['Volume']].reset_index()
    daily_volume = volumes.pivot('Date', 'Ticker', 'Volume')
    vol_mean = pd.DataFrame(daily_volume.mean(), columns=['vol_mean'])
    more_vola = vol_mean.sort_index(by='vol_mean', ascending=False)
    it_more_vola_tickers = more_vola[:12].index.tolist()  # todo: > valore medio

    # MA of volumes
    rvol = pd.rolling_mean(daily_volume[it_more_vola_tickers],60)
    rvol.plot()
    
    # find benchmark are for more volatile ETFs
    etfs, tickers = get_etf_tickers.get_etf_tickers()
    etfs2 = etfs.set_index('Reuters RIC (Italy)')
    etfs2.loc[it_more_vola_tickers]['Area Benchmark']
    
    # filter oldest AND more volatile
    df = pd.read_csv('csv/oldest_etfs.csv')
    it_oldest_tickers = df['Reuters RIC (Italy)'].tolist()
    
    
    
    
    
    
    
    
    
    
    