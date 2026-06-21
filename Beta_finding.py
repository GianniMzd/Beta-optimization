#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import scipy.stats as stats
# pyrefly: ignore [missing-import]
import yfinance as yf
from typing import Callable

# Set global seed for exact reproducibility across all random generations
np.random.seed(42)

# ==========================================
# 1. USER CONFIGURATION & MANDATES
# ==========================================

# Define your assets and their corresponding weights here.
# Ensure weights sum to 1.0

PORTFOLIO = {
    'AAPL': 0.40,  
    'MSFT': 0.30,  
    'GOOG': 0.20,  
    'JPM': 0.10    
}

START_DATE = "2021-06-01"
END_DATE = "2026-06-01"
M_PATHS = 50000  

# Exogenous Risk Constraints
VAR_LIMIT_DOLLARS = 50         
PORTFOLIO_VALUE = 600          
CONFIDENCE_LEVEL = 0.99   # For VaR(99%) —Basel III Regulations—            

# ==========================================
# 2. DATA ACQUISITION & PROCESSING
# ==========================================

def fetch_portfolio_returns(portfolio_dict, start_date, end_date):

    tickers = list(portfolio_dict.keys())
    weights = np.array(list(portfolio_dict.values()))
    
    # Fetch historical daily close prices
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.dropna() 
    
    # Calculate daily arithmetic returns
    asset_returns = data.pct_change().dropna()
    
    # Generate the historical portfolio returns time series
    historical_returns = asset_returns.dot(weights).to_numpy()
    
    return historical_returns

# ==========================================
# 3. RISK METRICS EVALUATION
# ==========================================

def calculate_var_cvar(sim_returns, alpha):

    # 1. Convert returns to losses (Profits become negative, losses become positive)
    losses = -sim_returns
    
    # 2. Sort from smallest to largest (Biggest losses go to the END of the array)
    sorted_losses = np.sort(losses) 
    
    # 3. Find the index for the tail cutoff (e.g., the 99% mark)
    k = int(np.floor(alpha * len(sorted_losses)))
    
    # 4. VaR is the exact loss at the cutoff
    var = sorted_losses[k]
    
    # 5. CVaR is the average of everything AFTER the cutoff (the worst scenarios)
    cvar = np.mean(sorted_losses[k:])
    
    return var, cvar

# ==========================================
# 4. MONTE CARLO SIMULATIONS
# ==========================================

def run_simulations(historical_returns, m_paths):

    n_days = len(historical_returns)
    
    # --- KERNEL DENSITY ESTIMATOR (KDE) ---
    kde = stats.gaussian_kde(historical_returns, bw_method='silverman')
    h = kde.factor * np.std(historical_returns)
    
    # Hierarchical sampling for KDE
    random_indices = np.random.choice(n_days, size=m_paths, replace=True)
    xi = historical_returns[random_indices]
    z = np.random.normal(0, 1, m_paths)
    sim_returns_kde = xi + h * z
    
    return sim_returns_kde

# ==========================================
# 5. MAX BETA OPTIMIZATION ENGINE
# ==========================================

def determine_optimal_beta(var_limit: float, compute_var_func: Callable[[float], float]) -> float:
    
    # Using the bisection algorithm

    beta_low = 1e-4   # Lowered floor so the algorithm can achieve very small limits (like $50)
    beta_high = 3.5  
    
    tol_beta = 1e-6   
    tol_dollar = 1.5 
    
    error = float('inf')
    iterations = 0
    max_iterations = 1000 
    
    while (beta_high - beta_low) > tol_beta and iterations < max_iterations:
        iterations += 1
        beta_mid = (beta_low + beta_high) / 2.0
        
        # Call the wrapper function for the current guess
        var_sim = compute_var_func(beta_mid)
        error = var_sim - var_limit
        
        if abs(error) < tol_dollar:
            break
        elif error > 0:
            beta_high = beta_mid  # Too Risky
        else:
            beta_low = beta_mid   # Too Conservative
            
    return (beta_low + beta_high) / 2.0

# ==========================================
# 6. EXECUTION 
# ==========================================

if __name__ == "__main__":
    
    # 1. Fetch Data
    hist_returns = fetch_portfolio_returns(PORTFOLIO, START_DATE, END_DATE)
    
    # 2. Simulate 50,000 paths ONCE to establish the base benchmark tail
    base_sim_kde = run_simulations(hist_returns, M_PATHS)
    
    # 3. Print the base benchmark metrics (Beta = 1.0)
    base_var, base_cvar = calculate_var_cvar(base_sim_kde, alpha=CONFIDENCE_LEVEL)
    print(f"Base Portfolio VaR (99%): {base_var*100:.2f}% (${base_var * PORTFOLIO_VALUE:,.2f})")
    print("-" * 60)

    # 4. Define the specific Wrapper for the optimizer
    def optimize_target_wrapper(beta_guess: float) -> float:

        # Scale the simulated benchmark returns by Beta
        scaled_sim_returns = beta_guess * base_sim_kde
        
        # Feed into your specific risk function
        var_pct, cvar_pct = calculate_var_cvar(scaled_sim_returns, alpha=CONFIDENCE_LEVEL)
        
        # Your function already returns a positive percentage (e.g., 0.045 for a 4.5% loss)
        # Convert it to absolute dollars
        var_dollar = PORTFOLIO_VALUE * var_pct
        return var_dollar

    # 5. Execute the Optimization
    optimal_beta = determine_optimal_beta(
        var_limit=VAR_LIMIT_DOLLARS, 
        compute_var_func=optimize_target_wrapper
    )
    
    print(f"Final Calculated Optimal Beta: {optimal_beta:.4f}")
    
    # Bonus: Show the CVaR at this exact Max Beta threshold
    final_var_99, final_cvar_99 = calculate_var_cvar(optimal_beta * base_sim_kde, alpha=0.99)
    final_var_975, final_cvar_975 = calculate_var_cvar(optimal_beta * base_sim_kde, alpha=0.975)
    
    print(f"At Optimal Beta, VaR (99%) is: {final_var_99*100:.2f}% (${final_var_99 * PORTFOLIO_VALUE:,.2f})")
    print(f"At Optimal Beta, CVaR (99%) is: {final_cvar_99*100:.2f}% (${final_cvar_99 * PORTFOLIO_VALUE:,.2f})")
    print(f"At Optimal Beta, VaR (97.5%) is: {final_var_975*100:.2f}% (${final_var_975 * PORTFOLIO_VALUE:,.2f})")
    print(f"At Optimal Beta, CVaR (97.5%) is: {final_cvar_975*100:.2f}% (${final_cvar_975 * PORTFOLIO_VALUE:,.2f})")