# import urllib2
import re
# from bs4 import BeautifulSoup
import pandas as pd
# import pandas.io.data
from pandas_datareader import data as pdr

# import fix_yahoo_finance as yf
# yf.pdr_override()

CSV_DIR = './csv/'


def get_one_ticker(ticker, start="1990-01-01", end="2019-11-01"):
    df = pdr.get_data_yahoo(ticker, start, end)
    return df


def get_tickers(ticker_list, start="1990-01-01", end="2019-11-01"):
    df = pd.DataFrame()
    for ticker in ticker_list:
        df[ticker] = get_one_ticker(ticker, start=start, end=end)['Adj Close']
    return df


def save_one_ticker(ticker, start="1990-01-01", end="2019-11-01"):
    # f = pandas.io.data.DataReader(ticker, "yahoo", start="1980/1/1")
    f = pdr.get_data_yahoo(ticker, start, end)
    # print(f.iloc[:1].to_string())
    # extract start date and store in a dict
    # log_dict[etf] = str(f.first_valid_index())
    # save quotes to CSV
    f.to_csv(CSV_DIR + ticker + '.csv')
    return 0


def get_etf_tickers():
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
            # { ticker1: [name1, isin1], ticker2: ...}
            etfs[ticker] = [name, isin]

    return etfs


def get_etf_tickers_2():
    # read from borsaitaliana.it official ETF list in xls.
    # returns a DataFrame containing ISIN, Name, Trading Code,
    # Reuters Code (Yahoo code), Bloomberg code and Area of ETF

    df = pd.read_excel('infoproviders.xlsx', header=8)
    return df
