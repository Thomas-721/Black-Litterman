from utils import config
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import base64
import requests
import pandas as pd
from pathlib import Path


class DataDownloader:
    """
    Base class for downloading and saving data.
    """
    
    def save_data(self, data, file_name):
        data.index = pd.to_datetime(data.index)
        data.to_csv(Path('data') / file_name)


class PriceVolumeDownloader(DataDownloader):
    """
    Downloads close price and volume data from Yahoo Finance.
    """
    
    def download(self, symbols):
        from_date = config.start_date
        to_date = config.end_date
        new_data = yf.download(symbols, start=from_date, end=to_date, progress=True)[['Close', 'Volume']]
        new_data.index = pd.to_datetime(new_data.index).strftime('%Y-%m-%d')
        
        benchmark_data = yf.download('SPY', start=from_date, end=to_date, progress=True)['Close']
        benchmark_data.index = pd.to_datetime(benchmark_data.index).strftime('%Y-%m-%d')

        self.save_data(new_data['Close'], 'price.csv')
        self.save_data(new_data['Volume'], 'volume.csv')
        self.save_data(benchmark_data, 'benchmark.csv')


class ShortInterestDownloader(DataDownloader):
    """
    Downloads short interest data from FINRA API.
    """
    
    def get_access_token(self):
        credentials = f"{config.finra_client_id}:{config.finra_client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        token_url = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials"
        headers = {"Authorization": f"Basic {encoded_credentials}", }
        response = requests.post(token_url, headers=headers)
        return response.json().get('access_token')

    def get_short_interest_helper(self, date, tickers, access_token):
        start, end = date
        data_url = "https://api.finra.org/data/group/otcmarket/name/consolidatedShortInterest"
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        payload = {
            "dateRangeFilters": [{"startDate": start, "endDate": end, "fieldName": "settlementDate"}],
            "domainFilters": [{"fieldName": "symbolCode", "values": tickers}],
            "limit": 100000,
        }

        res = requests.post(data_url, json=payload, headers=headers)
        if res.status_code == 200 and res.text:
            return (pd.DataFrame(res.json())
                   .rename(columns={"settlementDate": "Date", "symbolCode": "Symbol"})
                   [["Date", "Symbol", "currentShortPositionQuantity"]])

    def download(self, symbols, freq=30):
        from_date = config.start_date
        to_date = config.end_date
        access_token = self.get_access_token()
        start_range = pd.date_range(from_date, to_date, freq=f"{freq}D")
        date_range = [(start.strftime("%Y-%m-%d"), 
                      (start + pd.offsets.Day(freq - 1)).strftime("%Y-%m-%d"))
                     for start in start_range]

        with ThreadPoolExecutor(max_workers=20) as executor:
            res_list = list(tqdm(executor.map(lambda date: self.get_short_interest_helper(date, symbols, access_token), 
                                            date_range), total=len(date_range)))

        final_df = (pd.concat(res_list)
                    .pivot(index='Date', columns='Symbol', values='currentShortPositionQuantity')
                    .sort_index())
        self.save_data(final_df, 'short_interest.csv')


class MarketCapDownloader(DataDownloader):
    """
    Downloads market capitalization data from Financial Modeling Prep API.
    """
    
    def __init__(self):
        super().__init__()
        self.api_key = config.fmp_api_key
        self.base_url = "https://financialmodelingprep.com/api/v3/historical-market-capitalization/"

    def download(self, symbols):
        from_date = config.start_date
        to_date = config.end_date
        all_data = []
        for symbol in tqdm(symbols, desc="Downloading Market Cap"):
            url = f"{self.base_url}{symbol}?from={from_date}&to={to_date}&apikey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200 and response.text:
                data = response.json()
                df = (pd.DataFrame(data)
                     [['date', 'marketCap']]
                     .rename(columns={'date': 'Date', 'marketCap': symbol})
                     .assign(Date=lambda x: pd.to_datetime(x['Date']).dt.strftime('%Y-%m-%d'))
                     .set_index('Date'))
                all_data.append(df)

        if all_data:
            final_df = pd.concat(all_data, axis=1).sort_index()
            self.save_data(final_df, 'market_cap.csv')