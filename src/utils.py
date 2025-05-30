from pathlib import Path
import requests
import time
import pandas as pd


def fetch_and_save_stock_data(symbol: str, apikey: str, output_dir: Path, days: int = 1000):
    """
    Fetches the daily time series for a given stock symbol and saves the last N days to CSV.

    Parameters:
    - symbol: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'MBG.DEX')
    - apikey: Your Alpha Vantage API key
    - output_dir: Path object to the directory where the CSV will be saved
    - days: Number of most recent days to keep (default: 1000)
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={apikey}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"Error fetching data for {symbol}: {data.get('Note') or data.get('Error Message') or 'Unknown error'}")
        return

    # Parse and format the data
    timeseries = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(timeseries, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.columns = ["open", "high", "low", "close", "volume"]
    df_last_n = df.tail(days)

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{symbol}_last_{days}_days.csv"
    df_last_n.to_csv(csv_path)
    print(f"Saved data for {symbol} to {csv_path}")


def fetch_multiple_stocks(symbols: list, apikey: str, output_dir: Path, days: int = 1000):
    for symbol in symbols:
        fetch_and_save_stock_data(symbol, apikey, output_dir, days)
        time.sleep(12)  # To respect Alpha Vantage free-tier rate limits (5 calls per minute)
