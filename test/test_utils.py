import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

from src.utils import *

def test_fetch_and_save_stock_data(tmp_path):
    symbol = "TEST"
    apikey = "dummy"
    output_dir = tmp_path
    days = 2
    # Mocked API response
    mock_response = {
        "Time Series (Daily)": {
            "2024-06-01": {
                "1. open": "100",
                "2. high": "110",
                "3. low": "90",
                "4. close": "105",
                "5. volume": "1000"
            },
            "2024-05-31": {
                "1. open": "95",
                "2. high": "105",
                "3. low": "85",
                "4. close": "100",
                "5. volume": "1500"
            }
        }
    }
    with patch('utils.requests.get') as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_response)
        fetch_and_save_stock_data(symbol, apikey, output_dir, days)
        csv_path = output_dir / f"{symbol}_last_{days}_days.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path, index_col=0)
        assert len(df) == 2
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

def test_fetch_multiple_stocks(tmp_path):
    symbols = ["A", "B"]
    apikey = "dummy"
    output_dir = tmp_path
    days = 1
    # Patch fetch_and_save_stock_data to avoid actual API calls and file writes
    with patch('utils.fetch_and_save_stock_data') as mock_fetch:
        fetch_multiple_stocks(symbols, apikey, output_dir, days)
        assert mock_fetch.call_count == len(symbols)
        for i, symbol in enumerate(symbols):
            args, kwargs = mock_fetch.call_args_list[i]
            assert args[0] == symbol
            assert args[1] == apikey
            assert args[2] == output_dir
            assert args[3] == days
