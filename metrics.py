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

def get_sharp_ratio(arr: np.ndarray) -> float:
    ann_vol_ = arr.std() * np.sqrt(TRADING_DAYS)
    total_ret_ = arr.mean() * TRADING_DAYS
    return (total_ret_ - np.log(1.05)) / ann_vol_


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
    asset_ret_arr = df['asset_ret'].to_numpy()
    benchmark_ret_arr = df['benchmark_ret'].to_numpy()
    asset_max_dd = get_max_dd(asset_ret_arr)
    benchmark_max_dd = get_max_dd(benchmark_ret_arr)
    ann_asset_vol = get_ann_vol(asset_ret_arr)
    cgar = get_cgar(asset_ret_arr)
    calmar = get_calmar(cgar, asset_max_dd)
    asset_sharp = get_sharp_ratio (asset_ret_arr)
    asset_sortino = get_sortino_ratio(asset_ret_arr)    
    beta, alpha, r_2, information_ratio = get_r_pow_2(asset_ret_arr, benchmark_ret_arr)

    print (beta)
    print (alpha)
    print (r_2)
    print (information_ratio)
if __name__ == '__main__':
    df = pd.read_csv("out.csv")
    get_all_metrics(df)
    
