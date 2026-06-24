#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import scipy.stats as stats
# pyrefly: ignore [missing-import]
import scipy.optimize as optimize
# pyrefly: ignore [missing-import]
import yfinance as yf
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Set global seed for exact reproducibility
np.random.seed(42)

# ==========================================
# 1. USER CONFIGURATION & MANDATES
# ==========================================
TICKERS = ['TSLA', 'AIR.PA', 'GOOGL', 'AAPL', 'AF.PA', 'RACE.PA', 'BMW.DE', 'LVMH.PA', 'GLE.PA', 'GS', 'JPM', 'MSFT', 'GD', 'BA']
BENCHMARK = '^GSPC'  

# OUT-OF-SAMPLE BACKTEST TIMELINES
START_DATE = "2021-06-01"
SPLIT_DATE = "2024-06-01" # The wall between Train (In-Sample) and Test (Out-of-Sample)
END_DATE = "2026-06-01"

M_PATHS = 50000  

# Exogenous Risk Constraints
VAR_LIMIT_DOLLARS = 37.5        
PORTFOLIO_VALUE = 500          
CONFIDENCE_LEVEL = 0.99           
MAX_WEIGHT_PER_ASSET = 0.20 
MIN_WEIGHT_PER_ASSET = 0.015 

# ==========================================
# 2. DATA ACQUISITION & PARTITIONING
# ==========================================
def fetch_and_split_data(tickers, benchmark, start_date, split_date, end_date):
    all_tickers = tickers + [benchmark]
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
    
    data = data.dropna(axis=1, how='all')
    valid_tickers = [t for t in tickers if t in data.columns]
    data = data.dropna(axis=0, how='any')
    
    if len(data) == 0:
        raise ValueError("CRITICAL ERROR: No overlapping historical data found.")
    
    returns = data.pct_change().dropna()
    
    # SPLIT DATA: Train vs Test
    train_returns = returns.loc[:split_date]
    test_returns = returns.loc[split_date:]
    
    # Calculate Empirical Beta STRICTLY on Training Data
    train_benchmark_var = train_returns[benchmark].var()
    train_betas = train_returns[valid_tickers].apply(
        lambda col: col.cov(train_returns[benchmark]) / train_benchmark_var
    ).to_numpy()
    
    return train_returns[valid_tickers], train_returns[benchmark], test_returns[valid_tickers], test_returns[benchmark], train_betas, valid_tickers

# ==========================================
# 3. RISK METRICS EVALUATION
# ==========================================
def calculate_var_cvar(sim_returns, alpha):
    losses = -sim_returns
    sorted_losses = np.sort(losses) 
    k = int(np.floor(alpha * len(sorted_losses)))
    var = sorted_losses[k]
    cvar = np.mean(sorted_losses[k:])
    return var, cvar

# ==========================================
# 4. DETERMINISTIC MONTE CARLO (KDE)
# ==========================================
def run_simulations_deterministic(historical_returns, m_paths, random_indices, z_shocks):
    kde = stats.gaussian_kde(historical_returns, bw_method='silverman')
    h = kde.factor * np.std(historical_returns)
    xi = historical_returns.iloc[random_indices] if isinstance(historical_returns, pd.Series) else historical_returns[random_indices]
    sim_returns_kde = xi + h * z_shocks
    return sim_returns_kde

# ==========================================
# 5. MAX BETA OPTIMIZATION ENGINE (TRAIN ONLY)
# ==========================================
def maximize_portfolio_beta(train_asset_returns, train_betas):
    n_assets = len(train_betas)
    n_days = len(train_asset_returns)
    
    locked_indices = np.random.choice(n_days, size=M_PATHS, replace=True)
    locked_z_shocks = np.random.normal(0, 1, M_PATHS)
    
    def objective_function(weights):
        portfolio_beta = np.dot(weights, train_betas)
        return -portfolio_beta 
    
    def var_constraint(weights):
        port_hist_returns = train_asset_returns.dot(weights).to_numpy()
        sim_p = run_simulations_deterministic(port_hist_returns, M_PATHS, locked_indices, locked_z_shocks)
        var_pct, _ = calculate_var_cvar(sim_p, alpha=CONFIDENCE_LEVEL)
        var_dollar = PORTFOLIO_VALUE * var_pct
        return VAR_LIMIT_DOLLARS - var_dollar

    sum_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    risk_constraint = {'type': 'ineq', 'fun': var_constraint}
    
    bounds = tuple((MIN_WEIGHT_PER_ASSET, MAX_WEIGHT_PER_ASSET) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    print("Executing SLSQP solver on IN-SAMPLE data to find optimal weights...\n")
    result = optimize.minimize(
        objective_function, initial_weights, method='SLSQP', bounds=bounds, 
        constraints=[sum_constraint, risk_constraint], options={'disp': True, 'ftol': 1e-6}
    )
    return result.x

# ==========================================
# 6. OUT-OF-SAMPLE VISUALIZATION
# ==========================================
def export_oos_visualizations(optimal_weights, valid_tickers, test_asset_returns, test_benchmark_returns):
    print("\nGenerating Out-of-Sample Performance Dashboard...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Out-of-Sample Backtest: {SPLIT_DATE} to {END_DATE}", fontsize=16, fontweight='bold')

    # Calculate OOS returns
    oos_port_returns = test_asset_returns.dot(optimal_weights)
    cum_port = (1 + oos_port_returns).cumprod() - 1
    cum_bench = (1 + test_benchmark_returns).cumprod() - 1

    # Panel 1: Trajectory
    axes[0].plot(cum_port.index, cum_port * 100, label='Optimized Portfolio (Locked Weights)', color='indigo', linewidth=2)
    axes[0].plot(cum_bench.index, cum_bench * 100, label='S&P 500 Benchmark', color='gray', linestyle='--', linewidth=1.5)
    axes[0].set_title("True Out-of-Sample Trajectory (Unseen Data)", fontweight='bold')
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].legend(loc='upper left')

    # Panel 2: OOS Drawdown Comparison
    roll_max_port = (1 + oos_port_returns).cumprod().cummax()
    drawdown_port = ((1 + oos_port_returns).cumprod() / roll_max_port) - 1
    roll_max_bench = (1 + test_benchmark_returns).cumprod().cummax()
    drawdown_bench = ((1 + test_benchmark_returns).cumprod() / roll_max_bench) - 1

    axes[1].fill_between(drawdown_port.index, drawdown_port * 100, 0, color='indigo', alpha=0.3, label='Portfolio Drawdown')
    axes[1].plot(drawdown_bench.index, drawdown_bench * 100, color='gray', linestyle='--', label='Benchmark Drawdown')
    axes[1].set_title("Out-of-Sample Drawdown Profile", fontweight='bold')
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(loc='lower left')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("oos_backtest_dashboard.png", dpi=300, bbox_inches='tight')
    print("[SUCCESS] OOS Dashboard exported as 'oos_backtest_dashboard.png'")

# ==========================================
# 7. EXECUTION 
# ==========================================
if __name__ == "__main__":
    
    # 1. Fetch & Split Data
    train_assets, train_bench, test_assets, test_bench, train_betas, valid_tickers = fetch_and_split_data(
        TICKERS, BENCHMARK, START_DATE, SPLIT_DATE, END_DATE)
    
    # 2. Optimize STRICTLY on Training Data
    optimal_weights = maximize_portfolio_beta(train_assets, train_betas)
    
    print("\n" + "=" * 60)
    print(f"LOCKED PORTFOLIO WEIGHTS (TRAINED: {START_DATE} to {SPLIT_DATE})")
    print("=" * 60)
    for ticker, weight in zip(valid_tickers, optimal_weights):
        print(f"{ticker:<5}: {weight*100:>6.2f}%")
        
    # 3. Test on Out-Of-Sample Data
    oos_port_returns = test_assets.dot(optimal_weights)
    oos_port_beta = oos_port_returns.cov(test_bench) / test_bench.var()
    
    print("\n" + "=" * 60)
    print(f"OUT-OF-SAMPLE RESULTS ({SPLIT_DATE} to {END_DATE})")
    print("=" * 60)
    print(f"Target In-Sample Beta     : {np.dot(optimal_weights, train_betas):.4f}")
    print(f"Realized OOS Beta         : {oos_port_beta:.4f}")
    
    total_oos_return = (1 + oos_port_returns).prod() - 1
    total_bench_return = (1 + test_bench).prod() - 1
    print(f"Total OOS Portfolio Return: {total_oos_return*100:.2f}%")
    print(f"Total OOS Benchmark Return: {total_bench_return*100:.2f}%")
    print("=" * 60)
    
    # 4. Export OOS Visualizations
    export_oos_visualizations(optimal_weights, valid_tickers, test_assets, test_bench)