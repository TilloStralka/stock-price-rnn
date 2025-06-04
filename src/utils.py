from pathlib import Path
import requests
import time
import pandas as pd
from scipy.signal import find_peaks



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

def list_csv_files(folder_path: Path) -> list[str]:
    """
    Returns a list of all CSV file names (not paths) in the given folder.
    
    Args:
        folder_path (Path): The path to the folder to search.

    Returns:
        List[str]: List of CSV file names (with extension).
    """
    return [f.name for f in folder_path.glob("*.csv")]

def load_csvs_as_dfs(file_names: list[str], folder_path: Path) -> dict[str, pd.DataFrame]:
    """
    Reads each CSV file from the given list and returns a dictionary of DataFrames.

    Args:
        file_names (List[str]): List of CSV file names.
        folder_path (Path): Path to the folder containing the CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys as clean base names and values as DataFrames.
    """
    dataframes = {}
    for file_name in file_names:
        # Suggest using snake_case version of filename without '.csv'
        base_name = file_name.replace(".csv", "").replace("-", "_").replace(" ", "_").lower()
        df = pd.read_csv(folder_path / file_name)
        dataframes[f"df_{base_name}"] = df
    return dataframes


def add_moving_average(df, column='close', window=20):
    """
    Funktion gibt mir den moving average vom closing wert der jeweiligen aktie zurück 
    """
    df[f'sma_{window}'] = df[column].rolling(window=window).mean()
    return df


def add_local_peaks(df, column='close', distance=20):
    """
    Findet local peaks in einem zeitraum von 20 Tagen was bei einer gesammt betrachtung von 1000 tagen angemessen ist? 
    """
    peaks, _ = find_peaks(df[column], distance=distance)
    df['is_peak'] = False
    df.loc[df.index[peaks], 'is_peak'] = True
    return df

