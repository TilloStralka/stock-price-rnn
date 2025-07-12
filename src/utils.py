from pathlib import Path
import requests
import time
from typing import List
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt


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

def load_csvs_as_dfs(file_names: List[str], folder_path: Path) -> dict[str, pd.DataFrame]:
    """
    Reads each CSV file from the given list as time series and returns a dictionary of DataFrames.

    Assumes the first (unnamed) column contains date values.

    Args:
        file_names (List[str]): List of CSV file names.
        folder_path (Path): Path to the folder containing the CSV files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys as clean base names and values as time-indexed DataFrames.
    """
    dataframes = {}
    for file_name in file_names:
        # Create a snake_case base name from the file name
        base_name = file_name.replace(".csv", "").replace("-", "_").replace(" ", "_").lower()

        # Read CSV: use first column (index 0) as datetime index
        df = pd.read_csv(folder_path / file_name, parse_dates=[0], index_col=0)

        # Optional: rename index name to 'Date' for clarity
        df.index.name = "Date"

        # Store with key like "df_filename"
        dataframes[f"df_{base_name}"] = df

    return dataframes


def add_moving_average(df, column='close', window=20):
    """
    Funktion gibt mir den moving average vom closing wert der jeweiligen aktie zurück 
    """
    df[f'sma_{window}'] = df[column].rolling(window=window).mean()
    return df


def add_ema(df, column='close', span=20):
    """
    Fügt einer DataFrame-Spalte einen Exponential Moving Average (EMA) hinzu.

    Parameter:
    df (pd.DataFrame): Der DataFrame, dem die EMA-Spalte hinzugefügt werden soll.
    column (str): Der Name der Spalte, auf die der EMA berechnet wird (Standard: 'close').
    span (int): Die Länge des EMA-Fensters (Standard: 20).

    Rückgabe:
    pd.DataFrame: Der DataFrame mit einer neuen Spalte 'ema_{span}', die den EMA enthält.
    """
    # Berechne den Exponential Moving Average (EMA) für die angegebene Spalte
    df[f'ema_{span}'] = df[column].ewm(span=span, adjust=False).mean()
    return df

def add_local_peaks(df, column='close', distance=20):
    """
    Findet lokale Hochpunkte (Peaks) in einer bestimmten Spalte des DataFrames und markiert sie.

    Parameter:
    df (pd.DataFrame): Der DataFrame, in dem Peaks gesucht werden.
    column (str): Die Spalte, in der Peaks gesucht werden (Standard: 'close').
    distance (int): Mindestanzahl an Datenpunkten zwischen zwei Peaks (Standard: 20).

    Rückgabe:
    pd.DataFrame: Der DataFrame mit einer neuen booleschen Spalte 'is_peak', die True an lokalen Hochpunkten ist.
    """
    # Finde die Indizes lokaler Hochpunkte (Peaks) in der angegebenen Spalte
    peaks, _ = find_peaks(df[column], distance=distance)
    # Erstelle eine neue Spalte, die standardmäßig False ist
    df['is_peak'] = False
    # Setze die Einträge an den Peak-Indizes auf True
    df.loc[df.index[peaks], 'is_peak'] = True
    return df

def add_local_valleys(df, column='close', distance=20):
    """
    Findet lokale Tiefpunkte (Valleys) in einer bestimmten Spalte des DataFrames und markiert sie.

    Parameter:
    df (pd.DataFrame): Der DataFrame, in dem Valleys gesucht werden.
    column (str): Die Spalte, in der Valleys gesucht werden (Standard: 'close').
    distance (int): Mindestanzahl an Datenpunkten zwischen zwei Valleys (Standard: 20).

    Rückgabe:
    pd.DataFrame: Der DataFrame mit einer neuen booleschen Spalte 'is_valley', die True an lokalen Tiefpunkten ist.
    """
    # Finde die Indizes lokaler Tiefpunkte (Valleys) durch Invertierung der Werte
    valleys, _ = find_peaks(-df[column], distance=distance)
    # Erstelle eine neue Spalte, die standardmäßig False ist
    df['is_valley'] = False
    # Setze die Einträge an den Valley-Indizes auf True
    df.loc[df.index[valleys], 'is_valley'] = True
    return df


def calculate_statistics(df, column='close'):
    """
    Berechnet grundlegende statistische Kennzahlen für eine Spalte.
    """
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'variance': df[column].var()
    }

def add_lowess(df, column='close', frac=0.05):
    smoothed = lowess(df[column], df.index, frac=frac, return_sorted=False)
    df['lowess'] = smoothed
    return df

def add_savgol_filter(df, column='close', window_length=21, polyorder=2):
    df['savgol'] = savgol_filter(df[column], window_length=window_length, polyorder=polyorder)
    return df

def decompose_seasonality(df, column='close', period=252):
    """
    Zerlegt eine Zeitreihe in Trend-, saisonale und Restkomponenten mittels STL (Seasonal-Trend Decomposition using LOESS).

    Parameter:
    df (pd.DataFrame): DataFrame mit der Zeitreihe.
    column (str): Name der Spalte, die zerlegt werden soll (Standard: 'close').
    period (int): Saisonalitätsperiode, z.B. 252 für ein Börsenjahr.

    Rückgabe:
    pd.DataFrame: DataFrame mit zusätzlichen Spalten 'trend', 'seasonal' und 'resid'.
    """
    # Führe die STL-Zerlegung durch, um Trend, Saisonalität und Residuum zu extrahieren
    stl = STL(df[column], period=period, robust=True)
    result = stl.fit()
    df['trend'] = result.trend
    df['seasonal'] = result.seasonal
    df['resid'] = result.resid
    return df

def analyze_volume_peaks(df, volume_threshold_factor=1.5):
    """
    Markiert Tage mit ungewöhnlich hohem Handelsvolumen im Vergleich zum 20-Tage-Durchschnitt.

    Parameter:
    df (pd.DataFrame): DataFrame mit Volumendaten.
    volume_threshold_factor (float): Schwellenwert-Faktor für außergewöhnlich hohes Volumen (Standard: 1.5).

    Rückgabe:
    pd.DataFrame: DataFrame mit neuen Spalten 'volume_sma_20' und 'high_volume'.
    """
    # Berechne den gleitenden 20-Tage-Durchschnitt des Volumens
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    # Markiere Tage, an denen das Volumen das 1.5-fache des Durchschnitts überschreitet
    df['high_volume'] = df['volume'] > df['volume_sma_20'] * volume_threshold_factor
    return df



def plot_price_with_indicators(df, symbol='Aktie', save_path=None, show=True):
    """
    Erstellt einen Plot mit Close-Preis, gleitenden Durchschnitten, Peaks, Valleys und Handelsvolumen.
    Optional wird der Plot gespeichert.

    Parameter:
    df (pd.DataFrame): DataFrame mit Kurs- und Volumendaten sowie Indikator-Spalten.
    symbol (str): Name des Wertpapiers für die Plot-Beschriftung.
    save_path (str or Path): Wenn gesetzt, wird der Plot an diesem Pfad gespeichert.
    show (bool): Ob der Plot angezeigt werden soll (default: True).

    Rückgabe:
    None
    """
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Preis-Plot
    ax[0].plot(df.index, df['close'], label='Close', color='black')
    if 'sma_20' in df.columns:
        ax[0].plot(df.index, df['sma_20'], label='SMA 20', linestyle='--')
    if 'ema_20' in df.columns:
        ax[0].plot(df.index, df['ema_20'], label='EMA 20', linestyle='--', color='orange')
    if 'lowess' in df.columns:
        ax[0].plot(df.index, df['lowess'], label='Lowess', color='green', alpha=0.7)
    if 'savgol' in df.columns:
        ax[0].plot(df.index, df['savgol'], label='Savitzky-Golay', color='purple', alpha=0.7)

    if 'is_peak' in df.columns:
        ax[0].scatter(df.index[df['is_peak']], df['close'][df['is_peak']], color='red', label='Peaks')
    if 'is_valley' in df.columns:
        ax[0].scatter(df.index[df['is_valley']], df['close'][df['is_valley']], color='blue', label='Valleys')

    ax[0].set_title(f'{symbol} Preisverlauf & Indikatoren')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].bar(df.index, df['volume'], color='gray', alpha=0.6)
    if 'high_volume' in df.columns:
        ax[1].bar(df.index[df['high_volume']], df['volume'][df['high_volume']], color='red', label='High Volume')
        ax[1].legend()

    ax[1].set_ylabel('Volumen')
    ax[1].set_xlabel('Datum')
    ax[1].grid(True)

    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_trend_seasonal_resid(
    df: pd.DataFrame,
    symbol: str = "Asset",
    save_path: Path | str | None = None,
    show: bool = True,
) -> None:
    """
    Plot the trend, seasonal, and residual components of a time–series
    decomposition.

    If *save_path* is provided, the graphic is written to the **same folder**
    as the original chart but with “_trend” appended to the file name
    (e.g. ``apple.png`` → ``apple_trend.png``).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns ``trend``, ``seasonal``,
        and ``resid`` indexed by date.
    symbol : str, default "Asset"
        Label used in the plot title.
    save_path : pathlib.Path | str | None, default None
        Reference chart path.  The new file will be saved alongside it with
        “_trend” appended before the extension.  If *None*, the plot is not
        saved.
    show : bool, default True
        Whether to display the figure in the current session.  If *False*,
        the figure is closed after saving.

    Returns
    -------
    None
    """
    # Ensure the required columns are present
    required = {"trend", "seasonal", "resid"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"DataFrame is missing columns: {', '.join(missing)}")

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df.index, df["trend"], label="Trend", linewidth=1.2)
    ax.plot(df.index, df["seasonal"], label="Seasonal", linestyle="--")
    ax.plot(df.index, df["resid"], label="Residuals", linestyle=":")
    ax.set_title(f"{symbol} – Trend / Seasonal / Residual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Build the new save‑path with “_trend” appended before the suffix
    if save_path is not None:
        save_path = Path(save_path)
        trend_path = save_path.with_name(f"{save_path.stem}_trend{save_path.suffix}")
        fig.savefig(trend_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
