import urllib2
import re

from bs4 import BeautifulSoup
import pandas.io.data
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

CSV_DIR = './csv/'
    
    
def get_etf_tickers_teleborsa():
    # Scrapes list of ETF from teleborsa. Returns a dict with tickers as keys and name
    # and ISIN as values.

    url = 'http://www.teleborsa.it/Quotazioni/ETF'
    page = urllib2.urlopen(url)
    soup = BeautifulSoup(page.read())

    pat = '(\w+)-(\w+)-\w+$'
    etfs = {}

    for link in soup.find_all('a'):
        item = str(link.get('href'))
        if '/etf/' in item:
            name = link.string
            match = re.search(pat, item)  # ticker, isin
            ticker = match.group(1)
            isin = match.group(2)
            etfs[ticker] = [name, isin]  # { ticker1: [name1, isin1], ticker2: ...}

    return etfs


def get_etf_tickers():
    # read from borsaitaliana.it official ETF list in xls.
    # returns a DataFrame containing ISIN, Name, Trading Code,
    # Reuters Code (Yahoo code), Bloomberg code and Area of ETF

    df = pd.read_excel('infoproviders.xls', header=7)
    tickers_json_str = df['Reuters RIC (Italy)'].to_json(None, orient='records')
    tickers = json.loads(tickers_json_str)
    return df, tickers


def get(tickers, start, end):
    def data(ticker):
        try:
            return pandas.io.data.DataReader(ticker, 'yahoo', start, end)
        except (IOError, UnicodeEncodeError) as e:
            print ticker + ' :' + str(e)
    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker','Date'])


def load(tickers):
    def data(ticker):
        try:
            return pd.read_csv(CSV_DIR + ticker + '.csv', index_col=0)
        except IOError as e:
            print ticker + '.csv: ' + str(e)
    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker','Date'])


def render_scatter_plot(data, x_stock_name, 
                        y_stock_name, xlim=None, ylim=None):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.scatter(data[x_stock_name], data[y_stock_name])
    if xlim is not None: ax.set_xlim(xlim)
    ax.autoscale(False)
    ax.vlines(0, -10, 10)
    ax.hlines(0, -10, 10)
    ax.plot((-10, 10), (-10, 10))
    ax.set_xlabel(x_stock_name)
    ax.set_ylabel(y_stock_name)


if __name__ == '__main__':
    '''
    import json
    # etfs = get_etf_tickers()

    
    # # write etfs to json file
    # with open('etfs.json', 'w') as fp:
    #     json.dump(etfs, fp, indent=4)
    # fp.close()

    # # read etfs from json file
    # with open('etfs.json', 'r') as fp:
    #     etfs = json.load(fp)
    # fp.close()

    # # Displays a dict sorted alphabetically on keys
    # for key in sorted(etfs):
    #     print "%s: %s" % (key, etfs[key])

    # #tickers = list(etfs.iterkeys())
    # tickers = etfs.keys()

    etfs, tickers = get_etf_tickers()

    # Create folder to store CSVs

    import os
    try:
        os.mkdir(CSV_DIR)
    except OSError as e:
        pass

    # Download quotes and track info on ETFs
    log_dict = {}
    err_dict = {}

    for etf in tickers:
        print etf
    # for i in range(20):
    #     etf = tickers[i]
    #     print i, tickers[i]

        try:
            # f = pandas.io.data.DataReader(etf+'.mi', "yahoo", start="1980/1/1")
            f = pandas.io.data.DataReader(etf, "yahoo", start="1980/1/1")
            print f.iloc[:1].to_string()
            # extract start date and store in a dict
            log_dict[etf] = str(f.first_valid_index())
            # save quotes to CSV
            f.to_csv(CSV_DIR + etf + '.csv')
        except (IOError, UnicodeEncodeError) as e:
            print etf + ' :' + str(e)
            # store error in a dict
            err_dict[etf] = str(e)

    # Open a log file to track info on ETF (start date)
    with open(CSV_DIR + 'log_yahoo.txt', 'w') as logfile:
        json.dump(log_dict, logfile, indent=4)
    logfile.close()

    # Open a log file to track Yahoo quote download errors
    with open(CSV_DIR + 'err_yahoo.txt', 'w') as errfile:
        json.dump(err_dict, errfile, indent=4)
    errfile.close()

    print '========== END ========='
    print 'ETFs downloaded: ' + str(len(log_dict))
    print 'ETFs NOT downloaded: ' + str(len(err_dict))

    # for i in range(10):
    #     print i, tickers[i]
    #     try:
    #         f = pandas.io.data.DataReader(tickers[i]+'.mi', "yahoo", start="1980/1/1")
    #         print f.iloc[:1].to_string()
    #         # print f.iloc[:1]
    #         # f.loc['2006-11-23':'2006-11-27']
    #
    #         # # Write all the quotes of current ticker to file
    #         f.to_csv(tickers[i]+'.csv')
    #         # # Alternative method:
    #         # with open(tickers[i]+'.txt', 'w') as fp:
    #         #     print >> fp, f.iloc[:].to_string()
    #         # fp.close()
    #         #
    #     except IOError as e:
    #         print tickers[i] + ' :' + str(e)
    
    

    # --Analyse log file for start date, writing only oldest tickers to a file

    with open('log_yahoo.txt', 'r') as fp:
        myjson = json.load(fp)
    fp.close()
    df = pd.DataFrame.from_dict(myjson, orient='index')

    df.columns = ['start_date']
    df.sort_index(by='start_date')
    # Filtering: start date before a certain date
    newdf = df[df.start_date <= '2007-01-01 00:00:00'].sort_index(by='start_date')
    
    # printing more info on filtered ETFs
    largedf = pd.DataFrame()
    for item in newdf.index:
        # etfs[etfs['Reuters RIC (Italy)'] == 'XYP1.MI']        
        largedf = largedf.append(etfs[etfs['Reuters RIC (Italy)'] == item])
    # adding start dates to ETF info
    left = largedf.set_index('Reuters RIC (Italy)')
    right = newdf
    result = pd.concat([left, right], axis=1)
    # writing results to file
    result.to_csv(CSV_DIR + 'oldest_etfs.csv')
    
    
    

    # ---- GMR seekingalpha
    
    ticker_list = ['MDY','IEV','EEM','ILF','EPP','EDV','SHY']
    
    for etf in ticker_list:
        print etf
        try:
            # f = pandas.io.data.DataReader(etf+'.mi', "yahoo", start="1980/1/1")
            f = pandas.io.data.DataReader(etf, "yahoo", start="1980/1/1")
            print f.iloc[:1].to_string()
            # extract start date and store in a dict
            log_dict[etf] = str(f.first_valid_index())
            # save quotes to CSV
            f.to_csv(CSV_DIR + etf + '.csv')
        except (IOError, UnicodeEncodeError) as e:
            print etf + ' :' + str(e)
            # store error in a dict
            err_dict[etf] = str(e)

    # Open a log file to track info on ETF (start date)
    with open(CSV_DIR + 'log_yahoo_GMR.txt', 'w') as logfile:
        json.dump(log_dict, logfile, indent=4)
    logfile.close()

    # Open a log file to track Yahoo quote download errors
    with open(CSV_DIR + 'err_yahoo_GMR.txt', 'w') as errfile:
        json.dump(err_dict, errfile, indent=4)
    errfile.close()

    print '========== END ========='
    print 'ETFs downloaded: ' + str(len(log_dict))
    print 'ETFs NOT downloaded: ' + str(len(err_dict))

    '''
    
    # Adding volume information to etfs data
    # run in csv folder
    CSV_DIR = 'csv/'
    start_dates = []
    volumes = []
    etfs, tickers = get_etf_tickers()
    for item in tickers:
        try:
            filename = CSV_DIR + item + '.csv'
            etf = pd.read_csv(filename)
            volumes.append(etf[-252:].mean().Volume)
            start_dates.append(etf.Date[0])
        except (IOError, UnicodeEncodeError) as e:
            print item + ' :' + str(e)
            volumes.append(np.nan)
            start_dates.append(np.nan)
    etfs['start_date'] = start_dates
    etfs['volumes'] = volumes
    
                
            
    
    
    
    
    