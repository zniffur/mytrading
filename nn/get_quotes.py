import json
import pandas as pd
import pandas_datareader.data as web
import datetime
import os


def get_etf_tickers_2():
    df = pd.read_csv('sp500_tickers.csv')
    return df


if __name__ == "__main__":
    # parse CSV for tickers
    df_tickers = get_etf_tickers_2()
    j_tickers = df_tickers['Symbol'].to_json(None, orient='records')
    tickers = json.loads(j_tickers)

    # tickers = tickers[0:9]  # for testing
    # Create folder to store CSVs
    # CSV_DIR = './csv/'
    CSV_DIR = './csvy/'  # yahoo
    try:
        os.mkdir(CSV_DIR)
    except OSError as e:
        pass

    # Download quotes and track info on ETFs
    log_dict = {}
    err_dict = {}
    end = datetime.datetime(2018, 5, 31)
    start = datetime.datetime(2013, 5, 31)  # 5y

    for index, etf in enumerate(tickers):
        print(etf + " (" + str(index + 1) + "/" + str(len(tickers)) + ")")
        try:
            # f = web.DataReader(str(etf), 'iex', start, end)  # iex
            f = web.DataReader(str(etf), 'yahoo', start, end)  # yahoo

            # extract start date and store in a dict
            log_dict[etf] = str(f.first_valid_index())  # yahoo
            # log_dict[etf] = f.iloc[0].name[1].strftime('%Y-%m-%d')  # morningstar
            # log_dict[etf] = f.index[0]  # iex

            # save quotes to CSV
            f.to_csv(CSV_DIR + etf + '.csv')
        except (IOError, UnicodeEncodeError, KeyError) as e:
            print(etf + ' :' + str(e))
            # store error in a dict
            err_dict[etf] = str(e)

    # Open a log file to track info on ETF (start date)
    with open(CSV_DIR + 'start_dates.log', 'w') as logfile:
        json.dump(log_dict, logfile, indent=4)
    logfile.close()

    # Open a log file to track Yahoo quote download errors
    with open(CSV_DIR + 'error.log', 'w') as errfile:
        json.dump(err_dict, errfile, indent=4)
    errfile.close()

    print('========== END =========')
    print('Total tickers: ' + str(len(tickers)))
    print('Tickers downloaded: ' + str(len(log_dict)))
    print('Tickers NOT downloaded: ' + str(len(err_dict)))
