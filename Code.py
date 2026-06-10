import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA COLLECTION & PREPROCESSING
# ==========================================

def get_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Drop assets that have more than 10% missing data
    missing_pct = data.isnull().mean()
    valid_tickers = missing_pct[missing_pct < 0.10].index
    data = data[valid_tickers]
    
    # Handle remaining missing data via forward/backward fill
    data = data.ffill().bfill()
    
    # Calculate Log Returns: R_t = ln(P(t) / P(t-1))
    log_returns = np.log(data / data.shift(1)).dropna()
    
    return log_returns

# ==========================================
# 2 & 3. KDE & RISK METRICS (VaR / CVaR)
# ==========================================

def calculate_kde_cvar(weights, returns_matrix, alpha=0.95):

    # Compute historical portfolio returns for the given weight vector
    portfolio_returns = np.dot(returns_matrix, weights)
    
    # Fit Gaussian KDE using Silverman's rule for bandwidth selection
    kde = gaussian_kde(portfolio_returns, bw_method='silverman')
    
    # Step A: Expected Return (Mean of the portfolio returns)
    expected_return = np.mean(portfolio_returns)
    
    # Step B: Value at Risk (VaR)
    # Define an empirical search space for the quantile finding
    sorted_returns = np.sort(portfolio_returns)
    empirical_var = np.percentile(portfolio_returns, (1 - alpha) * 100)
    
    # Refine VaR using the KDE Cumulative Distribution Function (CDF)
    def cdf_objective(target_var):
        # Quantile definition: P(X <= VaR) = 1 - alpha
        prob, _ = quad(lambda x: kde.pdf(x), -np.inf, target_var, limit=100)
        return (prob - (1 - alpha)) ** 2
    
    # Minimize to find the exact threshold where CDF equals 1 - alpha
    res = minimize(cdf_objective, x0=empirical_var, method='Nelder-Mead')
    var_threshold = res.x[0]
    
    # Step C: Conditional Value at Risk (CVaR)
    # Integral expression: (1 / (1 - alpha)) * \int_{-inf}^{q} x * f(x) dx
    cvar_integral, _ = quad(lambda x: x * kde.pdf(x), -np.inf, var_threshold, limit=100)
    cvar = (1 / (1 - alpha)) * cvar_integral
    
    # CVaR is conventionally expressed as a positive loss value, 
    # but for consistent optimization math we keep its native negative magnitude.
    return expected_return, var_threshold, cvar

# ==========================================
# 4. PORTFOLIO OPTIMIZATION
# ==========================================

def optimize_portfolio(returns_matrix, target_lambda, alpha=0.99, min_weight=0.01, max_weight=0.05):
    
    num_assets = returns_matrix.shape[1] # take the number of assets (number of columns)
    
    # Initial guess: equally weighted portfolio
    init_weights = np.array([1.0 / num_assets] * num_assets)
    
    # Objective function to MINIMIZE (negative of our maximization objective)
    def objective_function(weights):
        exp_ret, _, cvar = calculate_kde_cvar(weights, returns_matrix, alpha)
        # We want to maximize return and minimize the absolute risk (negative cvar).
        # Thus, minimizing: - [lambda * exp_ret + (1 - lambda) * cvar]
        return -(target_lambda * exp_ret + (1 - target_lambda) * cvar)
    
    # Constraints: Fully invested (Sum of weights = 1)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Bounds: No short-selling and minimum allocation per asset for diversification
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
    
    # Run SLSQP Optimization
    result = minimize(
        objective_function, 
        init_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-5, 'maxiter': 100}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge perfectly: {result.message}")
        
    return result.x

# ==========================================
# 5. VALIDATION & BACKTESTING ENGINE
# ==========================================

def run_pipeline():
    # Asset Universe: A mix of 50 Liquid ETFs and Mega-cap equities 
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'GLD', 'SLV', 'TLT', 'LQD', 'HYG', 'VNQ',
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'UNH', 'JNJ', 'XOM',
        'JPM', 'V', 'PG', 'TSLA', 'HD', 'MA', 'ABBV', 'PFE', 'MRK', 'PEP',
        'KO', 'ORCL', 'TMO', 'AZN', 'CSCO', 'NKE', 'DIS', 'ADBE', 'CVX', 'AMD',
        'WMT', 'COST', 'BAC', 'MCD', 'LIN', 'ABT', 'CMCSA', 'VZ', 'AAPL', 'TXN'
    ]
    
    # Out-of-Sample Split Dates
    train_start, train_end = '2015-01-01', '2025-12-31'
    test_start, test_end = '2025-12-31', '2026-06-01'
    
    # Fetch Data
    all_returns = get_historical_data(tickers, start_date=train_start, end_date=test_end)
    
    # Split DataFrames
    train_returns = all_returns.loc[train_start:train_end].values
    test_returns = all_returns.loc[test_start:test_end].values
    tickers_final = all_returns.columns.tolist()
    
    # Balance weight between expected returns and CVaR protection
    optimal_weights = optimize_portfolio(train_returns, target_lambda=0.5, alpha=0.95, min_weight=0.01)
    
    # Create weight summary table
    weight_df = pd.DataFrame({'Asset': tickers_final, 'Weight': optimal_weights})
    weight_df = weight_df.sort_values(by='Weight', ascending=False)
    
    print("\nTop 10 Asset Allocations:")
    print(weight_df.head(10).to_string(index=False))
    
    # Evaluate performance on Out-of-Sample (Test) Data
    print(f"\n--- Phase 5: Out-of-Sample Validation ({test_start} to {test_end}) ---")
    
    # Equal Weight benchmark for comparison
    eq_weights = np.array([1.0 / len(tickers_final)] * len(tickers_final))
    
    opt_test_perf = np.dot(test_returns, optimal_weights)
    eq_test_perf = np.dot(test_returns, eq_weights)
    
    # Calculate cumulative metrics
    cum_opt = np.exp(np.cumsum(opt_test_perf)) - 1
    cum_eq = np.exp(np.cumsum(eq_test_perf)) - 1
    
    # Risk-free rate (assumed 0 for basic daily metrics)
    sharpe_opt = np.mean(opt_test_perf) / np.std(opt_test_perf) * np.sqrt(252)
    sharpe_eq = np.mean(eq_test_perf) / np.std(eq_test_perf) * np.sqrt(252)
    
    print(f"Optimized Portfolio Sharpe Ratio (Annualized): {sharpe_opt:.2f}")
    # Convert standard return directly to percentage for text presentation
    print(f"Optimized Portfolio Total Return: {cum_opt[-1]*100:.2f}%")
    print(f"Equal Weight Benchmark Sharpe Ratio: {sharpe_eq:.2f}")
    print(f"Equal Weight Benchmark Total Return: {cum_eq[-1]*100:.2f}%")
    
    # Plot performance results
    test_dates = all_returns.loc[test_start:test_end].index
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, cum_opt * 100, label=f'Optimized CVaR Portfolio (Sharpe: {sharpe_opt:.2f})', color='darkblue', lw=2)
    plt.plot(test_dates, cum_eq * 100, label=f'Equal Weight Benchmark (Sharpe: {sharpe_eq:.2f})', color='gray', linestyle='--')
    plt.title('Out-of-Sample Performance Comparison (Cumulative Returns %)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    run_pipeline()