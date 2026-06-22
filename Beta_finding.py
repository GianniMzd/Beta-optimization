#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import scipy.stats as stats
# pyrefly: ignore [missing-import]
import scipy.optimize as optimize
# pyrefly: ignore [missing-import]
import yfinance as yf
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns

# Set global seed for exact reproducibility across all random generations
np.random.seed(42)

# ==========================================
# 1. USER CONFIGURATION & MANDATES
# ==========================================
# Using the exact portfolio from your terminal
TICKERS = ['TSLA', 'AIR.PA', 'GOOGL', 'AAPL', 'AF.PA', 'RACE.PA', 'BMW.DE', 'LVMH.PA', 'GLE.PA', 'GS', 'JPM', 'MSFT', 'GD', 'BA']
BENCHMARK = '^GSPC'  # S&P 500 used to calculate the empirical Beta of each asset

START_DATE = "2021-06-01"
END_DATE = "2026-06-01"
M_PATHS = 50000  

# Exogenous Risk Constraints
VAR_LIMIT_DOLLARS = 50        
PORTFOLIO_VALUE = 500          
CONFIDENCE_LEVEL = 0.99           
MAX_WEIGHT_PER_ASSET = 0.125 # Forces diversification (Max 23.5% in a single asset)
MIN_WEIGHT_PER_ASSET = 0.01 # Forces diversification (Min 1.5% in a single asset)

# ==========================================
# 2. DATA ACQUISITION & BETA CALCULATION
# ==========================================
def fetch_asset_and_benchmark_data(tickers, benchmark, start_date, end_date):
    
    #Fetches historical data, drops delisted/invalid tickers, and calculates the empirical Beta for each surviving asset.
    all_tickers = tickers + [benchmark]
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
    
    # 1. Drop tickers that failed to download entirely (all NaNs)
    data = data.dropna(axis=1, how='all')
    
    # 2. Identify which requested tickers actually survived
    valid_tickers = [t for t in tickers if t in data.columns]
    failed_tickers = set(tickers) - set(valid_tickers)
    
    if failed_tickers:
        print(f"\n[WARNING] The following tickers failed to download and are excluded: {list(failed_tickers)}")
        
    # 3. Drop rows where any of the REMAINING valid assets are missing data
    data = data.dropna(axis=0, how='any')
    
    if len(data) == 0:
        raise ValueError("CRITICAL ERROR: No overlapping historical data found. Check your tickers.")
    
    # Calculate daily arithmetic returns
    returns = data.pct_change().dropna()
    
    asset_returns = returns[valid_tickers]
    benchmark_returns = returns[benchmark]
    
    # Calculate empirical Beta: Cov(R_i, R_m) / Var(R_m)
    benchmark_var = benchmark_returns.var()
    asset_betas = asset_returns.apply(lambda col: col.cov(benchmark_returns) / benchmark_var).to_numpy()
    
    print("\nEmpirical Asset Betas:")
    for t, b in zip(valid_tickers, asset_betas):
        print(f"  {t}: {b:.4f}")
        
    return asset_returns, benchmark_returns, asset_betas, valid_tickers

# ==========================================
# 3. RISK METRICS EVALUATION
# ==========================================
def calculate_var_cvar(sim_returns, alpha):
    
    # Calculates empirical VaR and CVaR from the tail of the loss distribution
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
    
    # Runs the KDE Smoothed Bootstrap using PRE-LOCKED random variables.
    kde = stats.gaussian_kde(historical_returns, bw_method='silverman')
    h = kde.factor * np.std(historical_returns)
    
    xi = historical_returns[random_indices]
    sim_returns_kde = xi + h * z_shocks
    return sim_returns_kde

# ==========================================
# 5. MAX BETA OPTIMIZATION ENGINE
# ==========================================
def maximize_portfolio_beta(asset_returns_df, asset_betas):
    
    # The multi-dimensional solver. Maximizes Beta subject to the KDE VaR limit
    n_assets = len(asset_betas)
    n_days = len(asset_returns_df)
    
    print(f"\nPre-generating {M_PATHS} random shocks to lock the KDE surface...")
    locked_indices = np.random.choice(n_days, size=M_PATHS, replace=True)
    locked_z_shocks = np.random.normal(0, 1, M_PATHS)
    
    # OBJECTIVE: Maximize Portfolio Beta 
    def objective_function(weights):
        portfolio_beta = np.dot(weights, asset_betas)
        return -portfolio_beta 
    
    # CONSTRAINT: KDE VaR must be <= Target Limit
    def var_constraint(weights):
        port_hist_returns = asset_returns_df.dot(weights).to_numpy()
        sim_p = run_simulations_deterministic(port_hist_returns, M_PATHS, locked_indices, locked_z_shocks)
        var_pct, _ = calculate_var_cvar(sim_p, alpha=CONFIDENCE_LEVEL)
        var_dollar = PORTFOLIO_VALUE * var_pct
        return VAR_LIMIT_DOLLARS - var_dollar

    sum_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    risk_constraint = {'type': 'ineq', 'fun': var_constraint}
    
    bounds = tuple((MIN_WEIGHT_PER_ASSET, MAX_WEIGHT_PER_ASSET) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    print("Executing SLSQP solver to maximize Beta under VaR constraint...\n")
    result = optimize.minimize(
        objective_function, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=[sum_constraint, risk_constraint],
        options={'disp': True, 'ftol': 1e-6}
    )
    
    return result.x, locked_indices, locked_z_shocks

# ==========================================
# 6. VISUALIZATION EXPORT
# ==========================================
def export_visualizations(optimal_weights, tickers, final_sim_returns, var_limit, port_value, asset_betas, asset_returns, benchmark_returns, alpha=0.99):
    
    # Generates and exports a 2x2 quantitative dashboard showing allocation, risk metrics, beta contributions, and historical performance.
    print("\nGenerating 2x2 Quantitative Dashboard...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("KDE Maximum Beta Optimization & Risk Dashboard", fontsize=18, fontweight='bold', y=0.95)
    
    # Filter out near-zero weights for cleaner charts
    mask = optimal_weights > 0.001
    filtered_weights = optimal_weights[mask]
    filtered_tickers = np.array(tickers)[mask]
    filtered_betas = asset_betas[mask]
    
    # -------------------------------------------------------------
    # Panel 1 (Top-Left): Optimal Portfolio Allocation
    # -------------------------------------------------------------
    colors = sns.color_palette("viridis", len(filtered_tickers))
    axes[0, 0].pie(filtered_weights, labels=filtered_tickers, autopct='%1.1f%%', 
                   startangle=140, colors=colors, textprops={'fontsize': 11})
    axes[0, 0].set_title("Optimal Weight Allocation", fontweight='bold', fontsize=14)
    
    # -------------------------------------------------------------
    # Panel 2 (Top-Right): Weighted Beta Contribution
    # -------------------------------------------------------------
    weighted_betas = filtered_weights * filtered_betas
    total_beta = np.sum(weighted_betas)
    
    sns.barplot(x=filtered_tickers, y=weighted_betas, ax=axes[0, 1], palette="magma")
    axes[0, 1].set_title(f"Marginal Beta Contribution (Total Portfolio $\\beta$ = {total_beta:.3f})", fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel("Weighted Beta ($w_i \\times \\beta_i$)")
    axes[0, 1].axhline(0, color='black', linewidth=1)
    
    # Annotate bars with exact numbers
    for i, p in enumerate(axes[0, 1].patches):
        axes[0, 1].annotate(f"{weighted_betas[i]:.3f}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=10, xytext=(0, 5), 
                            textcoords='offset points')

    # -------------------------------------------------------------
    # Panel 3 (Bottom-Left): KDE Loss Distribution & Risk Thresholds
    # -------------------------------------------------------------
    monetary_losses = -final_sim_returns * port_value
    sorted_losses = np.sort(monetary_losses)
    k = int(np.floor(alpha * len(sorted_losses)))
    var_dollar = sorted_losses[k]
    cvar_dollar = np.mean(sorted_losses[k:])
    
    sns.histplot(monetary_losses, bins=100, kde=True, ax=axes[1, 0], color="steelblue", 
                 stat="density", linewidth=0, alpha=0.4)
    
    axes[1, 0].axvline(x=var_dollar, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Achieved VaR (99%): ${var_dollar:,.2f}')
    axes[1, 0].axvline(x=cvar_dollar, color='red', linestyle='-', linewidth=2.5, 
                       label=f'CVaR (Expected Shortfall): ${cvar_dollar:,.2f}')
    axes[1, 0].axvline(x=var_limit, color='black', linestyle=':', linewidth=2.5, 
                       label=f'Hard Regulatory Limit: ${var_limit:,.2f}')
    
    axes[1, 0].set_title("Simulated KDE Tail Loss Distribution", fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel(f"Monetary Loss on ${port_value:,.2f} Portfolio")
    axes[1, 0].set_ylabel("Probability Density")
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].set_xlim(np.percentile(monetary_losses, 50), max(var_limit * 1.5, cvar_dollar * 1.2))

    # -------------------------------------------------------------
    # Panel 4 (Bottom-Right): In-Sample Historical Trajectory
    # -------------------------------------------------------------
    # Calculate historical trajectory of the optimal portfolio
    port_hist_returns = asset_returns.dot(optimal_weights)
    cum_port = (1 + port_hist_returns).cumprod() - 1
    cum_bench = (1 + benchmark_returns).cumprod() - 1
    
    axes[1, 1].plot(cum_port.index, cum_port * 100, label='Optimized Portfolio', color='indigo', linewidth=2)
    axes[1, 1].plot(cum_bench.index, cum_bench * 100, label='S&P 500 Benchmark', color='gray', linestyle='--', linewidth=1.5)
    
    axes[1, 1].set_title("Historical Trajectory vs Benchmark", fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel("Cumulative Return (%)")
    axes[1, 1].axhline(0, color='black', linewidth=1)
    axes[1, 1].legend(loc='upper left')
    
    # Format dates nicely
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    file_name = "portfolio_risk_dashboard.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] High-resolution quantitative dashboard exported as '{file_name}'")

# ==========================================
# 7. EXECUTION 
# ==========================================
if __name__ == "__main__":
    
    # 1. Fetch data and calculate empirical betas (Handles failed tickers)
    asset_returns, benchmark_returns, asset_betas, valid_tickers = fetch_asset_and_benchmark_data(TICKERS, BENCHMARK, START_DATE, END_DATE)
    
    # 2. Run the Optimization
    optimal_weights, locked_idx, locked_z = maximize_portfolio_beta(asset_returns, asset_betas)
    
    # 3. Print the optimal reallocation
    print("\n" + "=" * 60)
    print("MAX BETA PORTFOLIO ALLOCATION (RISK-BUDGETING)")
    print("=" * 60)
    for ticker, weight in zip(valid_tickers, optimal_weights):
        print(f"{ticker:<5}: {weight*100:>6.2f}%")
        
    print("-" * 60)
    
    # 4. Verify the final metrics of this newly allocated portfolio
    final_hist_returns = asset_returns.dot(optimal_weights).to_numpy()
    final_sim = run_simulations_deterministic(final_hist_returns, M_PATHS, locked_idx, locked_z)
    
    final_var_99, final_cvar_99 = calculate_var_cvar(final_sim, alpha=0.99)
    achieved_beta = np.dot(optimal_weights, asset_betas)
    
    print(f"Maximized Portfolio Beta   : {achieved_beta:.4f}")
    print(f"Final Simulated VaR (99%)  : {final_var_99*100:.2f}% (${final_var_99 * PORTFOLIO_VALUE:,.2f})")
    print(f"Final Simulated CVaR (99%) : {final_cvar_99*100:.2f}% (${final_cvar_99 * PORTFOLIO_VALUE:,.2f})")
    print("=" * 60)
    
    # 5. Export Visualizations
    export_visualizations(
        optimal_weights=optimal_weights,
        tickers=valid_tickers,
        final_sim_returns=final_sim,
        var_limit=VAR_LIMIT_DOLLARS,
        port_value=PORTFOLIO_VALUE,
        asset_betas=asset_betas,
        asset_returns=asset_returns,
        benchmark_returns=benchmark_returns,
        alpha=CONFIDENCE_LEVEL
    )