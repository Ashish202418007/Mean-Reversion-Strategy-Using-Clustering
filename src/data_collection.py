import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from utils.logger import setup_logger, log_function_call
warnings.filterwarnings('ignore')


load_dotenv("./.env")
TICKER_URL = os.getenv('NIFTY50_URL')
START_DATE = os.getenv('START_DATE')
END_DATE = os.getenv('END_DATE')

# logging
logger = setup_logger('download')
logger.info(f'Collecting data from {START_DATE} and {END_DATE}')

@log_function_call(get_logger=logger)
def fetch_single_ticker(ticker, start, end, interval, timeout=60, max_retries=3):
    """Fetch data for a single ticker with retries."""
    attempt = 0
    while attempt < max_retries:
        try:
            logger.info(f"Fetching data for {ticker}")
            data = yf.Ticker(ticker).history(start=start, end=end, interval=interval, timeout=timeout)

            if data is None or data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()

            data = data[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
            data['Ticker'] = ticker
            data.reset_index(inplace=True)
            logger.info(f"Successfully fetched data for {ticker}")
            return data

        except Exception as e:
            logger.error(f"Error fetching {ticker} on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(5)

    logger.error(f"Failed to fetch {ticker} after {max_retries} attempts")
    return pd.DataFrame()

@log_function_call(get_logger=logger)
def fetch_multiple_tickers_parallel(ticker_list, start, end, interval='1d', timeout=60, max_workers=10):
    """Fetch multiple tickers in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, ticker, start, end, interval, timeout): ticker for ticker in ticker_list}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    results.append(data)
                    logger.info(f"Data collected for {ticker}")
                else:
                    logger.warning(f"No data returned for {ticker}")
            except Exception as exc:
                logger.error(f"{ticker} generated an exception: {exc}")

    if results:
        final_df = pd.concat(results, ignore_index=True)
        logger.info(f"Fetched total {len(final_df)} rows across {len(results)} tickers.")
        return final_df
    else:
        logger.warning("No data fetched for any ticker.")
        return pd.DataFrame()

@log_function_call(get_logger=logger)
def main():
    try:
        logger.info("Fetching ticker list...")
        tickers = pd.read_html(TICKER_URL)[0]['Symbol'].tolist()
        # tickers = [ticker + '.NS' for ticker in tickers]

        if 'ETERNAL.NS' in tickers:
            index = tickers.index('ETERNAL.NS')
            tickers[index] = 'ZOMATO.NS'
            logger.info("Replaced ETERNAL.NS with ZOMATO.NS")

        logger.info(f"Fetching data from {START_DATE} to {END_DATE} for {len(tickers)} tickers.")

        data = fetch_multiple_tickers_parallel(tickers, interval='1d', start=START_DATE, end=END_DATE)
        if not data.empty:
            logger.info(f"Retrieved {len(data)} records successfully.")
            output_path = "./data/raw/NIFTY50data.csv"
            data.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}")
        else:
            logger.warning("No data returned.")

    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}")

if __name__ == "__main__":
    main()
