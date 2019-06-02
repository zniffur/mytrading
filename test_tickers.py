from tickers import get_one_ticker, get_tickers
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d")
now = '2019-05-24'


def test_one_ticker():
    ticker = 'SPY'
    df = get_one_ticker(ticker, start="1990-01-01", end="2019-11-01")
    assert (df.iloc[0, :].name is not None)


def test_multiple_tickers():
    ticker_list = ['SPY', 'IEF', 'VT', 'TLT', 'CSSPX.MI']
    df2 = get_tickers(ticker_list, start="1990-01-01", end="2019-11-01")
    assert (df2.loc[now] is not None)


"""
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

etfs = get_etf_tickers_2()
tickers_json_str = etfs['Reuters RIC (Italy)'].to_json(
    None, orient='records')
tickers = json.loads(tickers_json_str)

# Create folder to store CSVs
CSV_DIR = './csv/'
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
print 'Total ETFs: ' + str(len(tickers))
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
 """
