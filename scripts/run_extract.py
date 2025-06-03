
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd  # Nur nötig, wenn du direkt CSV speicherst
import logging

# Import your fetch function
from src.utils import fetch_multiple_stocks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Script started.")

# Automatisch den Projekt-Root finden, egal von wo das Skript ausgeführt wird
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

# Read environment variables
api_key = os.getenv("API_KEY")
db_user = os.getenv("DB_USER")
debug_mode = os.getenv("DEBUG", "false").lower() == "true"

def main():
    if not api_key:
        logging.error("API key is missing. Please set it in your .env file.")
        return

    # Define stocks and target folder
    stock_list = ["AAPL", "MSFT", "GOOGL", "MBG.DEX"]
    data_folder = Path(__file__).resolve().parents[1] / "data" / "raw"

    # Create folder if it doesn't exist
    data_folder.mkdir(parents=True, exist_ok=True)

    if debug_mode:
        logging.info(f"Debug mode is ON")
        logging.info(f"Saving data to: {data_folder.resolve()}")

    try:
        fetch_multiple_stocks(stock_list, api_key, data_folder)
        logging.info("ETL run completed successfully.")
    except Exception as e:
        logging.exception("ETL run failed.")

if __name__ == "__main__":
    main()
