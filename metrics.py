import pandas as pd
import numpy as np
import math
# Konstanten
RISK_FREE_RATE = 0.05 #ann%
TRADING_DAYS = 252

def get_max_dd(arr: np.ndarray) -> float:
    equity = np.cumsum(arr)
    maximum = np.maximum.accumulate(equity)
    runup =  maximum - equity
    return (1-np.exp(-runup.max()))


def get_ann_vol(arr: np.ndarray) -> float:
    return arr.std() * np.sqrt(TRADING_DAYS)

def get_rolling_vol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df['asset_rolling_vol'] = df['asset_ret'].rolling(window=window, min_periods=window).std() * np.sqrt(window)
    df['benchmark_rolling_vol'] = df['benchmark_ret'].rolling(window=window, min_periods=window).std() * np.sqrt(window)
    return df

def get_sharp_ratio(arr: np.ndarray) -> float:
    rolling_vol_ = arr.std() * np.sqrt(TRADING_DAYS)
    total_ret_ = arr.mean() * TRADING_DAYS
    return (total_ret_ - np.log(1.05) ) / rolling_vol_

def get_rolling_sharp(df: pd.DataFrame, window: int) -> pd.DataFrame:
    rf_daily = np.log(1 + RISK_FREE_RATE) / TRADING_DAYS
    
    #Asset calc
    asset_excess_returns = df['asset_ret'] - rf_daily
    asset_rolling_mean = asset_excess_returns.rolling(window=window, min_periods=window).mean()
    asset_rolling_std = df['asset_ret'].rolling(window=window, min_periods=window).std()
    df['asset_rolling_sharpe'] = (asset_rolling_mean / asset_rolling_std) * np.sqrt(TRADING_DAYS)
    
    #Benchmark calc
    benchmark_exess_ret = df['benchmark_ret'] - rf_daily
    bench_rolling_mean = benchmark_exess_ret.rolling(window=window, min_periods=window).mean()
    bench_rolling_std = df['benchmark_ret'].rolling(window=window, min_periods=window).std()
    df['benchmark_rolling_sharpe'] = (bench_rolling_mean / bench_rolling_std) * np.sqrt(TRADING_DAYS)

    return df


def get_sortino_ratio(arr: np.ndarray) -> float:
    total_ret_ = arr.mean() * TRADING_DAYS 
    downside = np.minimum(arr, 0.0)
    sigma = np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)
    return (total_ret_ - np.log(1.05)) / sigma

def get_cgar(arr: np.ndarray) -> float:
    t_ = arr.size / TRADING_DAYS
    total_ret = np.sum(arr) 
    return np.exp(total_ret / t_) - 1

def get_calmar(cgar: float, max_dd: float):
    return cgar/max_dd

def get_beta(asset: np.ndarray, benchmark:np.ndarray):
    cov_mat = np.cov(m=asset,y=benchmark)
    return cov_mat[0,1]/cov_mat[1,1]
    
def get_alpha(beta: float, asset: np.ndarray, benchmark:np.ndarray) -> float:
    asset_ret =  asset.mean() * TRADING_DAYS
    benchmark_ret =  benchmark.mean() * TRADING_DAYS
    rf = np.log(1.05)
    ea = asset_ret - rf
    eb = benchmark_ret - rf
    return ea -beta*eb

def get_r_pow_2(asset: np.ndarray, benchmark:np.ndarray) -> tuple[float, float, float, float]:#beta, alpha täglich, r², info ratio
    
    rf = np.log(1.05) / TRADING_DAYS
    ea = asset - rf
    eb = benchmark - rf
    cov_mat = np.cov(m=ea,y=eb)
    beta = cov_mat[0,1]/cov_mat[1,1]
    alpha = (ea.mean() -beta*eb.mean())
    y_hat = alpha + beta * eb
    epsilon = ea - y_hat
    r_2 = 1 - epsilon.var() / ea.var()
    information_ratio = (alpha / epsilon.std()) * np.sqrt(TRADING_DAYS)
    return (beta, alpha, r_2, information_ratio)

def get_all_metrics(df: pd.DataFrame):
    asset_ret_arr = df['asset_ret'].copy().to_numpy()
    benchmark_ret_arr = df['benchmark_ret'].copy().to_numpy()
    max_dd = get_max_dd(asset_ret_arr)
    cgar = get_cgar(asset_ret_arr)
    calmar = get_calmar(cgar, max_dd)
    sortino = get_sortino_ratio(asset_ret_arr)    
    beta, alpha, r_2, information_ratio = get_r_pow_2(asset_ret_arr, benchmark_ret_arr)

    return (max_dd, cgar, calmar, sortino, beta, alpha, r_2, information_ratio)
if __name__ == '__main__':
    df = pd.read_csv("out.csv")
    get_all_metrics(df)
    
