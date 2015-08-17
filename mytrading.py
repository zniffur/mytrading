__author__ = 'simo'

from pyalgotrade.tools import yahoofinance

if __name__ == '__main__':
    yahoofinance.download_daily_bars('orcl', 2000, 'orcl-2000.csv')
    pass