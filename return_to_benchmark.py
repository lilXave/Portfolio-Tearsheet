import pandas as pd
import yfinance as yf
import math
import numpy as np

def get_csv_data(start_date : str, benchmark : str, asset : str) -> pd.DataFrame :
    INTERVAL = "1d"
    asset_data = yf.Ticker(asset).history(start=start_date, interval=INTERVAL)
    benchmark_data = yf.Ticker(benchmark).history(start=start_date, interval=INTERVAL)
    asset_log_ret = np.log(asset_data['Close']).diff()
    asset_log_ret = asset_log_ret[~np.isnan(asset_log_ret)]
    benchmark_log_ret = np.log(benchmark_data['Close']).diff()
    benchmark_log_ret = benchmark_log_ret[~np.isnan(benchmark_log_ret)]
    
    assert np.issubdtype(benchmark_log_ret.dtype, np.floating), benchmark_log_ret.dtype
    assert np.isfinite(benchmark_log_ret).all()
    assert np.issubdtype(asset_log_ret.dtype, np.floating), asset_log_ret.dtype
    assert np.isfinite(asset_log_ret).all()

    return (pd.DataFrame({'asset_ret': asset_log_ret, 'benchmark_ret': benchmark_log_ret}))


def write_csv(df : pd.DataFrame) :
    df.to_csv(path_or_buf="out.csv")

if __name__ == '__main__':
    start_date = "2010-01-01"
    asset = "AAPL"
    benchmark = 'SPY'

    df = get_csv_data(start_date=start_date, asset=asset, benchmark=benchmark)

    write_csv(df)
    print("Done!")