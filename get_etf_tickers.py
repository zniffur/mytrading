from bs4 import BeautifulSoup
import urllib2
import re

def get_etf_tickers():
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
			etfs[ticker] = [name, isin]  #  { ticker1: [name1, isin1], ticker2: ...}

	return etfs			
	

if __name__ == "__main__":

	import pandas
	import pandas.io.data
	
	etfs = get_etf_tickers()
	
	tickers = list(etfs.iterkeys())
	for i in range(10):
		print i, tickers[i]
		try:
			f = pandas.io.data.DataReader(tickers[i]+'.mi', "yahoo", start="1980/1/1")
			# print f.iloc[:1].to_string()
			# print f.iloc[:1]
			f.head()
			# f.loc['2006-11-23':'2006-11-27']
		except IOError as e:
			print tickers[i] + ' :' + str(e)