import numpy as np
import math
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import io

# Part 1 - Input Data
def input_data(S_0=None,r=None,sigma=None,T=None,K=None,M=None):
    """
    Function to input necessary parameters for option pricing.
    If parameters are not provided, prompts user for input.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    
    Returns:
    - Tuple containing validated input parameters.
    """
    # We set it up like this to allow the user to put the arguments into the function directly or request input from them if they don't do so.
    if all(param is not None for param in [S_0, r, sigma, T, K, M]):
        return S_0, r, sigma, T, K, M
    
    while True:
        try:
            if S_0 is None:
                S_0 = float(input('Enter the initial stock price: \n'))
            if r is None:
                r = float(input('Enter the risk-free interest rate as a decimal (per annum): \n'))
            if sigma is None:
                sigma = float(input('Enter the volatility of the stock price as a decimal (per annum): \n'))
            if T is None:
                T = float(input('Enter the expiry time of the option in months: \n'))
            if K is None:
                K = float(input('Enter the strike price of the option: \n'))
            if M is None:
                M = int(input('Enter the number of periods in the binomial tree: \n'))

            if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                print('Error: Ensure all parameters are non-negative and logical: \n'
                    'risk-free interest rate should not take negative values,'
                    'expiry time should be greater than or equal to 0,'
                    'number of periods in the binomial tree should be greater than 0.')
            else:
                return S_0, r, sigma, T, K, M # Return values if they pass the check
            
        except ValueError:
            print('Error: Invalid input. Please enter numeric values.\n')

input_data(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50)

# Part 2 - The Binomial model
def binomial_CRR(S_0=None,r=None,sigma=None,T=None,K=None,M=None,call_option=None):
    """
    Implements the Binomial Cox-Ross-Rubinstein (CRR) model to price European options.
    If parameters are not provided, prompts user for input.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - Option price, mean stock price, second moment of stock price, computational time
    """
    if any(param is None for param in [S_0, r, sigma, T, K, M, call_option]):
        while True:
            try:
                S_0 = float(input('Enter the initial stock price: \n'))
                r = float(input ('Enter the risk-free interest rate as a decimal(per annum): \n'))
                sigma = float(input('Enter the volatility of the stock price as a decimal(per annum): \n'))
                T = float(input('Enter the expiry time of the option in months: \n'))
                K = float(input('Enter the strike price of the option: \n'))
                M = int(input('Enter the number of periods in the binomial tree: \n'))
                call_option = input('Enter True for a call option, False for a put option: \n').strip().lower()

                if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                    print('Error: Ensure all parameters are non-negative and logical: \n'
                        'risk-free interest rate should not take negative values,'
                        'expiry time should be greater than or equal to 0,'
                        'number of periods in the binomial tree should be greater than 0.')
                    
                if call_option not in ['true', 'false']:
                        print('Error: Option type must be "True" for a call or "False" for a put. \n')

                else:
                    return S_0, r, sigma, T, K, M  # Return values if they pass the check
            
            except ValueError:
                print('Error: Invalid input. Please enter numeric values.\n')

    start_time = time.time()

    T = T / 12  # Convert months to years
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    ps = (np.exp(r * dt) - d) / (u - d)
    
    if ps <= 0 or ps >= 1:
        print('Error: Model is not arbitrage-free.')
        return None, None
    
    V = np.zeros((M + 1, M + 1))
    stock_prices = np.zeros(M + 1)
    probabilities = np.zeros(M + 1)
    
    for j in range(M + 1):
        S = S_0 * u**j * d**(M - j)
        stock_prices[j] = S
        probabilities[j] = np.math.comb(M, j) * (ps**j) * ((1 - ps) ** (M - j))
        if call_option:
            V[j, M] = max(S - K, 0)
        else: 
            V[j, M] = max(K - S, 0)
    
    for i in range(M - 1, -1, -1):
        for j in range(i + 1):
            V[j, i] = np.exp(-r * dt) * (ps * V[j + 1, i + 1] + (1 - ps) * V[j, i + 1])
    
    mean_stock_price = np.sum(stock_prices * probabilities)
    second_moment_stock_price = np.sum((stock_prices ** 2) * probabilities)
    computational_time = time.time() - start_time

    print(f'Option Price: {V[0, 0]:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')
    

    return V[0, 0], mean_stock_price, second_moment_stock_price, computational_time

binomial_CRR(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)
binomial_CRR(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=False)

def binomial_CRR_optimised(S_0=None,r=None,sigma=None,T=None,K=None,M=None,call_option=None):
    """
    Implements an optimised Binomial Cox-Ross-Rubinstein (CRR) model to price European options.
    
    This function computes the option price using a binomial model while optimising computations.
    If parameters are not provided, prompts user for input.

    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - V: Computed option price
    - mean_stock_price: Expected stock price at maturity
    - second_moment_stock_price: Second moment of the stock price at maturity
    - computational_time: Time taken for computation (in seconds)
    """
    if any(param is None for param in [S_0, r, sigma, T, K, M, call_option]):
        while True:
            try:
                S_0 = float(input('Enter the initial stock price: \n'))
                r = float(input ('Enter the risk-free interest rate as a decimal(per annum): \n'))
                sigma = float(input('Enter the volatility of the stock price as a decimal(per annum): \n'))
                T = float(input('Enter the expiry time of the option in months: \n'))
                K = float(input('Enter the strike price of the option: \n'))
                M = int(input('Enter the number of periods in the binomial tree: \n'))
                call_option = input('Enter True for a call option, False for a put option: \n').strip().lower()

                if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                    print('Error: Ensure all parameters are non-negative and logical: \n'
                        'risk-free interest rate should not take negative values,'
                        'expiry time should be greater than or equal to 0,'
                        'number of periods in the binomial tree should be greater than 0.')
                    
                if call_option not in ['true', 'false']:
                        print('Error: Option type must be "True" for a call or "False" for a put. \n')

                else:
                    return S_0, r, sigma, T, K, M  # Return values if they pass the check
            
            except ValueError:
                print('Error: Invalid input. Please enter numeric values.\n')

    start_time = time.time()

    T = T / 12
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    ps=(np.exp(r * dt) - d)/(u - d)
    
    if ps<=0 or ps>=1:
        print('Error: Not arbitrage-free.')
        return None

    V = 0  # Running sum for the discounted expected payoff
    binomial_coeff = 1  # Initial binomial coefficient (C(M, 0))

    mean_stock_price = 0
    second_moment_stock_price = 0

    for j in range(M + 1):
        S_j = S_0 * (u**j) * (d**(M - j))  # Stock price at node (j, M)
        if call_option:
            h_S_j = max(S_j - K, 0)  # Call option payoff
        else:
            h_S_j = max(K - S_j, 0)  # Put option payoff

        probability = binomial_coeff * (ps**j) * ((1-ps)**(M-j))
        
        # Add to the running sum for the option value
        V += probability * h_S_j

        mean_stock_price += probability * S_j
        second_moment_stock_price += probability * (S_j**2)
        
        # Update binomial coefficient for the next iteration
        if j < M:
            binomial_coeff *= (M - j) / (j + 1)

    V *= np.exp(-r * T)  # Discount the expected payoff back to present value

    computational_time = time.time() - start_time

    print(f'Option Price: {V:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of the Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')

    return V, mean_stock_price, second_moment_stock_price, computational_time

binomial_CRR_optimised(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)
binomial_CRR_optimised(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=False)

def plot_computational_time_comparison(S_0, r, sigma, T, K, filename='computational_time_comparison.pdf'):
    """
    Compares the computational time of the standard and optimised Binomial CRR models for both call and put options.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - filename: Name of the file to save the plot
    """
    M_values = np.arange(2, 301, 4) # Define a linear range for M
    M_values = np.logspace(np.log10(M_values[0]), np.log10(M_values[-1]), num=len(M_values), dtype=int)  # Convert to log scale
    
    computational_times_crr_call = []
    computational_times_crr_optimised_call = []
    computational_times_crr_put = []
    computational_times_crr_optimised_put = []

    # Suppress print statements from other functions
    with contextlib.redirect_stdout(io.StringIO()):
        for M in M_values:
            _, _, _, time_crr_call = binomial_CRR(S_0, r, sigma, T, K, M, call_option=True)
            _, _, _, time_crr_optimised_call = binomial_CRR_optimised(S_0, r, sigma, T, K, M, call_option=True)
        
            _, _, _, time_crr_put = binomial_CRR(S_0, r, sigma, T, K, M, call_option=False)
            _, _, _, time_crr_optimised_put = binomial_CRR_optimised(S_0, r, sigma, T, K, M, call_option=False)
        
            computational_times_crr_call.append(time_crr_call)
            computational_times_crr_optimised_call.append(time_crr_optimised_call)
            computational_times_crr_put.append(time_crr_put)
            computational_times_crr_optimised_put.append(time_crr_optimised_put)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Call option plot
    axes[0].plot(M_values, computational_times_crr_call, label='Approximate Binomial CRR Using Matrix', marker='o', color='red')
    axes[0].plot(M_values, computational_times_crr_optimised_call, label='Approximate Binomial CRR Optimised', marker='s', color='blue')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Number of Steps (M) (log scale)')
    axes[0].set_ylabel('Computational Time (s) (log scale)')
    axes[0].set_title('Computational Time Comparison (Call Option)')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Put option plot
    axes[1].plot(M_values, computational_times_crr_put, label='Approximate Binomial CRR Using Matrix', marker='o', color='red')
    axes[1].plot(M_values, computational_times_crr_optimised_put, label='Approximate Binomial CRR Optimised', marker='s', color='blue')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Number of Steps (M) (log scale)')
    axes[1].set_ylabel('Computational Time (s) (log scale)')
    axes[1].set_title('Computational Time Comparison (Put Option)')
    axes[1].legend()
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()
    
    print(f'Plot saved as {filename}')

plot_computational_time_comparison(S_0=285, r=0.021, sigma=0.12, T=20, K=290, filename='computational_time_comparison.pdf')

# Part 3 - Verification
def black_scholes(S_0=None,r=None,sigma=None,T=None,K=None,call_option=None):
    """
    Implements the Black-Scholes formula for European options pricing.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - call_option: True for call option, False for put option
    
    Returns:
    - Option price, absolute error, mean stock price, second moment of stock price, computational time
    """
    start_time = time.time()

    T = T / 12
    
    # Calculate d1 and d2
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate option price based on type
    if call_option:
        price = S_0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
    
    # Calculate the mean and second moment of the stock price at time T
    mean_stock_price = S_0 * np.exp(r * T)
    second_moment_stock_price = S_0**2 * np.exp((2 * r + sigma**2) * T)
    absolute_error = 0

    computational_time = time.time() - start_time

    print(f'Option Price: {price:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of the Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')

    return price, absolute_error, mean_stock_price, second_moment_stock_price, computational_time

black_scholes(285,0.021,0.12,20,290,call_option=True)
black_scholes(285,0.021,0.12,20,290,call_option=False)

#Adding absolute error to output of binomial_CRR_optimised so tables are easier to make in later sections
def binomial_CRR_optimised_with_error(S_0, r, sigma, T, K, M, call_option):
    # Suppress print statements from other functions
    with contextlib.redirect_stdout(io.StringIO()):
        V, mean_stock_price, second_moment_stock_price, computational_time = binomial_CRR_optimised(S_0, r, sigma, T, K, M, call_option)
        black_scholes_price, _, _, _, _ = black_scholes(S_0, r, sigma, T, K, call_option)

    absolute_error = abs(V - black_scholes_price)

    return V, absolute_error, mean_stock_price, second_moment_stock_price, computational_time

binomial_CRR_optimised_with_error(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)
binomial_CRR_optimised_with_error(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=False)

def prepare_plot_data(S_0, r, sigma, T, K, call_option, pricing_function):
    """
    Prepares data for plotting the convergence of binomial model prices to the Black-Scholes price.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - call_option: True for a call option, False for a put option
    - pricing_function: Function used to compute binomial option prices
    
    Returns:
    - M_values: Array of M values used for the binomial model
    - prices_even: List of option prices for even M values
    - prices_odd: List of option prices for odd M values
    """  
    M_values = np.arange(2, 301, 2)  # M values from 2 to 500 (even)
    prices_even = []
    prices_odd = []

    # Calculate option prices for both even and odd M using any of our pricing functions
    for M in M_values:
        price_even, _, _, _, _ = pricing_function(S_0, r, sigma, T, K, M, call_option)
        price_odd, _, _, _, _ = pricing_function(S_0, r, sigma, T, K, M + 1, call_option)
        
        # Collect prices for plotting
        prices_even.append(price_even)
        prices_odd.append(price_odd)

    return M_values, prices_even, prices_odd

def plot_convergence_to_black_scholes(S_0, r, sigma, T, K, pricing_function=None, filename=None):
    """
    Generates and plots the convergence of binomial tree models to Black-Scholes prices
    for both call and put options, displaying them side by side in a grid.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - pricing_function: Function used to compute binomial option prices
    - filename: Name of the file to save the plot (optional)
    
    Returns:
    - None (Displays the plot and optionally saves it to a file)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns
    
    with contextlib.redirect_stdout(io.StringIO()):
        exact_price_call, _, _, _, _ = black_scholes(S_0, r, sigma, T, K, call_option=True)
        exact_price_put, _, _, _, _ = black_scholes(S_0, r, sigma, T, K, call_option=False)

    for i, (call_option, exact_price) in enumerate([(True, exact_price_call), (False, exact_price_put)]):
        M_values, prices_even, prices_odd = prepare_plot_data(S_0, r, sigma, T, K, call_option, pricing_function)

        # Select subplot
        ax = axes[i]

        # Plot binomial tree prices for even and odd M
        ax.plot(M_values, prices_even, label='Binomial Price (M even)', color='red', marker='o', markersize=3, linestyle='--')
        ax.plot(M_values, prices_odd, label='Binomial Price (M odd)', color='blue', marker='+', markersize=3, linestyle='--')
        ax.axhline(y=exact_price, color='grey', linestyle='--', label=f'Black-Scholes Price: {exact_price:.4f}')

        # Labels and title
        ax.set_xlabel('Number of Steps (M)')
        ax.set_ylabel('Option Price')
        ax.set_title(f'Convergence of Binomial Model to Black-Scholes Price ({ "Call" if call_option else "Put" } Option)')
        ax.legend()
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a filename is provided
    if filename:
        plt.savefig(filename, format='pdf', bbox_inches="tight")

    plt.show()

plot_convergence_to_black_scholes(S_0=285, r=0.021, sigma=0.12, T=20, K=290, pricing_function=binomial_CRR_optimised_with_error, filename = 'CRR_convergence.pdf')

# Part 4 - Investigating the choice of upward stock price movement
def plot_stock_price_distribution_with_cdf(S_0, r, sigma, T, M, filename='stock_distribution.pdf'):
    """
    Plots the stock price distribution (PDF & CDF) comparing the Binomial model and Black-Scholes model.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - M: Number of steps in the binomial model
    - filename: Name of the file to save the plot (optional)
    
    Returns:
    - None (Displays the plot and optionally saves it to a file)
    """
    T = T/12
    dt = T/M
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    ps = (np.exp(r * dt) - d) / (u - d)

    # Generate stock prices at maturity using vectorized operations
    j_vals = np.arange(M + 1)
    stock_prices_binomial = S_0 * (u**j_vals) * (d**(M - j_vals))

    # Compute binomial probabilities using the binomial PMF
    binomial_coeffs = np.array([np.math.comb(M, j) for j in j_vals])
    probabilities_binomial = binomial_coeffs * (ps**j_vals) * ((1 - ps)**(M - j_vals))
    probabilities_binomial /= np.sum(probabilities_binomial)  # Normalise

    # Generate Black-Scholes distribution
    stock_prices_bs = np.linspace(min(stock_prices_binomial), max(stock_prices_binomial), 1000)
    mean_log = np.log(S_0) + (r - 0.5 * sigma**2) * T
    std_dev = sigma * np.sqrt(T)
    pdf_bs = norm.pdf(np.log(stock_prices_bs), mean_log, std_dev) / stock_prices_bs  # Black-Scholes PDF
    cdf_bs = norm.cdf(np.log(stock_prices_bs), mean_log, std_dev)  # Black-Scholes CDF

    # Compute binomial cumulative probabilities (CDF)
    cdf_binomial = np.cumsum(probabilities_binomial)

    # Create subplots for PDF and CDF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot PDF (Probability Density Function)
    axes[0].hist(stock_prices_binomial, weights=probabilities_binomial, bins=25, alpha=0.6, density=True, label='Binomial PDF')
    axes[0].plot(stock_prices_bs, pdf_bs, label='Black-Scholes PDF', color='red', linewidth=2)
    axes[0].set_xlabel('Stock Price at Maturity')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Stock Price Distribution: Binomial vs Black-Scholes (PDF)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot CDF (Cumulative Density Function)
    axes[1].step(stock_prices_binomial, cdf_binomial, label='Binomial CDF', where='post', color='blue')
    axes[1].plot(stock_prices_bs, cdf_bs, label='Black-Scholes CDF', color='red', linewidth=2)
    axes[1].set_xlabel('Stock Price at Maturity')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('CDF Comparison: Binomial vs Black-Scholes')
    axes[1].legend()
    axes[1].grid(True)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()
    
    print(f'Plot saved as {filename}')

plot_stock_price_distribution_with_cdf(S_0=285, r=0.021, sigma=0.12, T=20, M=50)

def binomial_CRR_adapted_u_argument(S_0=None,r=None,sigma=None,T=None,K=None,M=None,u=None,call_option=None):
    """
    Computes the option price using the Binomial Cox-Ross-Rubinstein (CRR) model
    while taking the upward stock price movement factor (u) as an argument.
    If parameters are not provided, prompts user for input.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - u: Upward stock price movement factor
    - call_option: True for a call option, False for a put option
    
    Returns:
    - V: Computed option price
    - mean_stock_price: Expected stock price at maturity
    - second_moment_stock_price: Second moment of the stock price at maturity
    """
    if any(param is None for param in [S_0, r, sigma, T, K, M, call_option]):
        while True:
            try:
                S_0 = float(input('Enter the initial stock price: \n'))
                r = float(input ('Enter the risk-free interest rate (per annum): \n'))
                sigma = float(input('Enter the volatility of the stock price (per annum): \n'))
                T = float(input('Enter the expiry time of the option in months: \n'))
                K = float(input('Enter the strike price of the option: \n'))
                M = int(input('Enter the number of periods in the binomial tree: \n'))
                call_option = input('Enter True for a call option, False for a put option: \n').strip().lower()

                if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                    print('Error: Ensure all parameters are non-negative and logical: \n'
                        'risk-free interest rate should not take extreme negative values,'
                        'expiry time should be greater than or equal to 0,'
                        'number of periods in the binomial tree should be greater than 0.')
                    
                if call_option not in ['true', 'false']:
                        print('Error: Option type must be "True" for a call or "False" for a put. \n')

                else:
                    return S_0, r, sigma, T, K, M  # Return values if they pass the check
            
            except ValueError:
                print('Error: Invalid input. Please enter numeric values.\n')
    
    T = T / 12  # Convert time to years
    dt = T / M
    d = 1 / u 
    ps = (np.exp(r * dt) - d) / (u - d) 

    if ps <= 0 or ps >= 1:
        print(f"Error: Not arbitrage-free for u = {u:.4f}. Skipping this value.")
        return None, None, None

    V = 0  
    binomial_coeff = 1
    mean_stock_price = 0
    second_moment_stock_price = 0

    for j in range(M + 1):
        S_j = S_0 * (u**j) * (d**(M - j))  # Stock price at node (j, M)
        h_S_j = max(S_j - K, 0) if call_option else max(K - S_j, 0)  # Payoff
        
        probability = binomial_coeff * (ps**j) * ((1 - ps)**(M - j))
        V += probability * h_S_j
        mean_stock_price += probability * S_j
        second_moment_stock_price += probability * (S_j**2)
        
        if j < M:
            binomial_coeff *= (M - j) / (j + 1)

    V *= np.exp(-r * T)  # Discount to present value

    return V, mean_stock_price, second_moment_stock_price

def optimal_u(S_0, r, sigma, T, K, M, call_option):
    """
    Determines the optimal up-factor (u) for the binomial tree model by minimising the absolute error
    between the binomial model price and the Black-Scholes price.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - u_values: Array of u values tested
    - option_prices: Corresponding option prices for each u value
    - best_u: The optimal u that minimises the absolute error
    - best_price: The binomial price corresponding to best_u
    - absolute_error: The absolute error between best_price and Black-Scholes price
    - best_mean: Expected stock price at maturity for best_u
    - best_second_moment: Second moment of the stock price at maturity for best_u
    - total_computational_time: Time taken to compute the optimal u
    """
    start_time = time.time()

    T_years = T / 12
    dt = T_years / M
    min_u = np.exp(r * dt)*(1 + 1e-6)  # Ensure u > exp(r * dt)
    u_values = np.linspace(min_u, 1.04, 20000)

    with contextlib.redirect_stdout(io.StringIO()):
        # Calculate Black-Scholes price for reference
        black_scholes_price, _, _, _, _ = black_scholes(S_0, r, sigma, T, K, call_option)

        option_prices = []
        absolute_errors = []
        mean_stock_values = []
        second_moment_values = []

        for u in u_values:
            result = binomial_CRR_adapted_u_argument(S_0, r, sigma, T, K, M, u, call_option)
            if result[0] is not None:
                option_price, mean_stock, second_moment = result
                option_prices.append(option_price)
                absolute_errors.append(abs(option_price - black_scholes_price))
                mean_stock_values.append(mean_stock)
                second_moment_values.append(second_moment)
            else:
                option_prices.append(np.nan)
                absolute_errors.append(np.nan)
                mean_stock_values.append(np.nan)
                second_moment_values.append(np.nan)

        # Find the u that gives the minimum absolute error
        best_u_index = np.nanargmin(absolute_errors)
        best_u = u_values[best_u_index]
        best_price = option_prices[best_u_index]
        best_mean = mean_stock_values[best_u_index]
        best_second_moment = second_moment_values[best_u_index]

        # Calculate total computational time for the entire process
        total_computational_time = time.time() - start_time

        absolute_error = abs(best_price - black_scholes_price)

    return u_values, option_prices, best_u, best_price, absolute_error, best_mean, best_second_moment, total_computational_time, black_scholes_price

def plot_combined_option_price(S_0, r, sigma, T, K, M, filename='option_price_vs_u_combined.pdf'):
    """
    Generates and plots the relationship between the up-factor (u) and the option price
    for both call and put options, displaying them side by side in a single figure.
    
    This function computes the optimal up-factor (u) for call and put options separately,
    finds the corresponding binomial option prices, and plots the results.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - filename: Name of the file to save the plot (optional)
    
    Returns:
    - None (Displays the plot and optionally saves it to a file)
    """
    # Get plot data for call and put options
    u_values_call, prices_call, best_u_call, best_price_call, abs_error_call, best_mean_call, best_second_moment_call, computational_time, bs_price_call = optimal_u(S_0, r, sigma, T, K, M, call_option=True)
    u_values_put, prices_put, best_u_put, best_price_put, abs_error_put, best_mean_put, best_second_moment_put, computational_time, bs_price_put = optimal_u(S_0, r, sigma, T, K, M, call_option=False)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Call option plot
    axes[0].plot(u_values_call, prices_call, color='blue', label='Binomial Call Option Price')
    axes[0].axhline(y=bs_price_call, color='red', linestyle='--', label='True Call Price')
    axes[0].axvline(x=best_u_call, color='green', linestyle='--', label=f'Best u = {best_u_call:.4f}')
    axes[0].set_xlabel('Up Factor (u)')
    axes[0].set_ylabel('Option Price')
    axes[0].set_title('Call Option Price vs. Up Factor (u)')
    axes[0].legend()
    axes[0].grid(True)

    # Put option plot
    axes[1].plot(u_values_put, prices_put, color='blue', label='Binomial Put Option Price')
    axes[1].axhline(y=bs_price_put, color='red', linestyle='--', label='True Put Price')
    axes[1].axvline(x=best_u_put, color='green', linestyle='--', label=f'Best u = {best_u_put:.4f}')
    axes[1].set_xlabel('Up Factor (u)')
    axes[1].set_ylabel('Option Price')
    axes[1].set_title('Put Option Price vs. Up Factor (u)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()

    print(f'Plot saved as {filename}')

plot_combined_option_price(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50)

# Part 5 - Tilted Tree and Richardson Extrapolation
def binomial_CRR_exact(S_0=None,r=None,sigma=None,T=None,K=None,M=None,call_option=None):
    """
    Computes the option price using the exact Binomial Cox-Ross-Rubinstein (CRR) model.
    
    This function calculates the European option price using a binomial tree approach
    with an exact formula for the up (u) and down (d) movement factors.
    If parameters are not provided, prompts user for input.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - V: Computed option price
    - absolute_error: Absolute error between the computed price and Black-Scholes price
    - mean_stock_price: Expected stock price at maturity
    - second_moment_stock_price: Second moment of the stock price at maturity
    - computational_time: Time taken for computation (in seconds)
    """
    if any(param is None for param in [S_0, r, sigma, T, K, M, call_option]):
        while True:
            try:
                S_0 = float(input('Enter the initial stock price: \n'))
                r = float(input ('Enter the risk-free interest rate (per annum): \n'))
                sigma = float(input('Enter the volatility of the stock price (per annum): \n'))
                T = float(input('Enter the expiry time of the option in months: \n'))
                K = float(input('Enter the strike price of the option: \n'))
                M = int(input('Enter the number of periods in the binomial tree: \n'))
                call_option = input('Enter True for a call option, False for a put option: \n').strip().lower()

                if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                    print('Error: Ensure all parameters are non-negative and logical: \n'
                        'risk-free interest rate should not take extreme negative values,'
                        'expiry time should be greater than or equal to 0,'
                        'number of periods in the binomial tree should be greater than 0.')
                    
                if call_option not in ['true', 'false']:
                        print('Error: Option type must be "True" for a call or "False" for a put. \n')

                else:
                    return S_0, r, sigma, T, K, M  # Return values if they pass the check
            
            except ValueError:
                print('Error: Invalid input. Please enter numeric values.\n')

    start_time = time.time()

    T = T / 12
    dt = T / M
    beta = 0.5 * (np.exp(-r * dt) + np.exp((r + sigma**2) * dt))
    u = beta + np.sqrt(beta**2 - 1)
    d = beta - np.sqrt(beta**2 - 1)
    ps=(np.exp(r * dt) - d)/(u - d)
    
    if ps<=0 or ps>=1:
        print('Error: Not arbitrage-free.')
        return None

    V = 0  # Running sum for the discounted expected payoff
    binomial_coeff = 1  # Initial binomial coefficient (C(M, 0))

    mean_stock_price = 0
    second_moment_stock_price = 0

    for j in range(M + 1):
        S_j = S_0 * (u**j) * (d**(M - j))  # Stock price at node (j, M)
        if call_option:
            h_S_j = max(S_j - K, 0)  # Call option payoff
        else:
            h_S_j = max(K - S_j, 0)  # Put option payoff

        probability = binomial_coeff * (ps**j) * ((1-ps)**(M-j))
        
        # Add to the running sum for the option value
        V += probability * h_S_j

        mean_stock_price += probability * S_j
        second_moment_stock_price += probability * (S_j**2)
        
        # Update binomial coefficient for the next iteration
        if j < M:
            binomial_coeff *= (M - j) / (j + 1)

    V *= np.exp(-r * T)  # Discount the expected payoff back to present value

    computational_time = time.time() - start_time

    with contextlib.redirect_stdout(io.StringIO()):
        # Calculate the Black-Scholes price to compute the absolute error
        black_scholes_price, _, _, _, _ = black_scholes(S_0, r, sigma, T*12, K, call_option)

    absolute_error = abs(V - black_scholes_price)

    print(f'Option Price: {V:.4f}')
    print(f'Absolute Option Price Error: {absolute_error:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of the Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')

    return V, absolute_error, mean_stock_price, second_moment_stock_price, computational_time

binomial_CRR_exact(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)

def tilted_tree(S_0=None,r=None,sigma=None,T=None,K=None,M=None,call_option=None):
    """
    Computes the option price using a tilted binomial tree model.
    
    This function adjusts the up (u) and down (d) movement factors dynamically to better
    align the tree structure with the strike price, improving convergence properties.
    If parameters are not provided, prompts user for input.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - V: Computed option price
    - absolute_error: Absolute error between the computed price and Black-Scholes price
    - mean_stock_price: Expected stock price at maturity
    - second_moment_stock_price: Second moment of the stock price at maturity
    - computational_time: Time taken for computation (in seconds)
    """
    if any(param is None for param in [S_0, r, sigma, T, K, M, call_option]):
        while True:
            try:
                S_0 = float(input('Enter the initial stock price: \n'))
                r = float(input ('Enter the risk-free interest rate (per annum): \n'))
                sigma = float(input('Enter the volatility of the stock price (per annum): \n'))
                T = float(input('Enter the expiry time of the option in months: \n'))
                K = float(input('Enter the strike price of the option: \n'))
                M = int(input('Enter the number of periods in the binomial tree: \n'))
                call_option = input('Enter True for a call option, False for a put option: \n').strip().lower()

                if S_0 < 0 or r < 0 or sigma < 0 or T <= 0 or K < 0 or M < 0:
                    print('Error: Ensure all parameters are non-negative and logical: \n'
                        'risk-free interest rate should not take extreme negative values,'
                        'expiry time should be greater than or equal to 0,'
                        'number of periods in the binomial tree should be greater than 0.')
                    
                if call_option not in ['true', 'false']:
                        print('Error: Option type must be "True" for a call or "False" for a put. \n')

                else:
                    return S_0, r, sigma, T, K, M  # Return values if they pass the check
            
            except ValueError:
                print('Error: Invalid input. Please enter numeric values.\n')

    start_time = time.time()

    T = T / 12
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt) + (1 / M) * np.log(K / S_0))
    d = np.exp(-sigma * np.sqrt(dt) + (1 / M) * np.log(K / S_0))
    ps=(np.exp(r * dt) - d)/(u - d)
    
    if ps<=0 or ps>=1:
        print('Error: Not arbitrage-free.')
        return None

    V = 0  # Running sum for the discounted expected payoff
    binomial_coeff = 1  # Initial binomial coefficient (C(M, 0))

    mean_stock_price = 0
    second_moment_stock_price = 0

    for j in range(M + 1):
        S_j = S_0 * (u**j) * (d**(M - j))  # Stock price at node (j, M)
        if call_option:
            h_S_j = max(S_j - K, 0)  # Call option payoff
        else:
            h_S_j = max(K - S_j, 0)  # Put option payoff

        probability = binomial_coeff * (ps**j) * ((1-ps)**(M-j))
        
        # Add to the running sum for the option value
        V += probability * h_S_j

        mean_stock_price += probability * S_j
        second_moment_stock_price += probability * (S_j**2)
        
        # Update binomial coefficient for the next iteration
        if j < M:
            binomial_coeff *= (M - j) / (j + 1)

    V *= np.exp(-r * T)  # Discount the expected payoff back to present value

    computational_time = time.time() - start_time

    with contextlib.redirect_stdout(io.StringIO()):
        # Calculate the Black-Scholes price to compute the absolute error
        black_scholes_price, _, _, _, _ = black_scholes(S_0, r, sigma, T*12, K, call_option)

    absolute_error = abs(V - black_scholes_price)

    print(f'Option Price: {V:.4f}')
    print(f'Absolute Option Price Error: {absolute_error:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of the Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')

    return V, absolute_error, mean_stock_price, second_moment_stock_price, computational_time

tilted_tree(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)

def tilted_tree_with_richardson(S_0, r, sigma, T, K, M, call_option=True):
    """
    Computes the option price using a tilted binomial tree model with Richardson extrapolation.
    
    This function applies Richardson extrapolation to improve the accuracy of the binomial
    option pricing method by using two different step sizes (M and M/2) and combining the results.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for a call option, False for a put option
    
    Returns:
    - P_ex: Extrapolated option price
    - absolute_error: Absolute error between the computed price and Black-Scholes price
    - mean_stock_price: Expected stock price at maturity
    - second_moment_stock_price: Second moment of the stock price at maturity
    - computational_time: Time taken for computation (in seconds)
    """
    start_time = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        # Calculate option prices, mean stock price, and second moment using tilted_tree for M and M/2
        P_M, _, mean_M, moment_M, _ = tilted_tree(S_0, r, sigma, T, K, M, call_option)
        P_M_half, _, mean_M_half, moment_M_half, _ = tilted_tree(S_0, r, sigma, T, K, M // 2, call_option)
    
    if P_M is None or P_M_half is None:
        print("Error: Could not calculate option prices for both M and M/2.")
        return None

    # Apply Richardson extrapolation
    P_ex = 2 * P_M - P_M_half
    mean_stock_price = 2 * mean_M - mean_M_half
    second_moment_stock_price = 2 * moment_M - moment_M_half

    computational_time = time.time() - start_time 

    with contextlib.redirect_stdout(io.StringIO()):
        # Calculate the Black-Scholes price for reference
        black_scholes_price, _, _, _, _ = black_scholes(S_0, r, sigma, T, K, call_option)

    absolute_error = abs(P_ex - black_scholes_price)

    print(f'Option Price: {P_ex:.4f}')
    print(f'Absolute Option Price Error: {absolute_error:.4f}')
    print(f'Mean Stock Price: {mean_stock_price:.4f}')
    print(f'Second Moment of the Stock Price: {second_moment_stock_price:.4f}')
    print(f'Computational Time: {computational_time:.6f} seconds')

    return P_ex, absolute_error, mean_stock_price, second_moment_stock_price, computational_time

tilted_tree_with_richardson(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)

def generate_comparison_table(S_0, r, sigma, T, K, M, call_option=True, base_filename='comparison_table'):
    """
    Generate a table comparing different pricing models for a European option.
    
    This function computes option prices using multiple models (Black-Scholes, Binomial CRR,
    Optimal u, Tilted Tree, and Richardson Extrapolation) and stores the results in a pandas
    DataFrame. The results include option price, absolute error, mean stock price, second moment,
    and computational time. The function also exports the table to a LaTeX file.
    
    Parameters:
    - S_0: Initial stock price
    - r: Risk-free interest rate (per annum)
    - sigma: Volatility of the stock price (per annum)
    - T: Expiry time of the option in months
    - K: Strike price of the option
    - M: Number of periods in the binomial tree
    - call_option: True for call option, False for put option
    - base_filename: Base filename for exporting the table to LaTeX format
    
    Returns:
    - A pandas DataFrame containing the results.
    """
    # Collect results from all models
    results = {
        "Model": [],
        "Option Price": [],
        "Absolute Error": [],
        "Mean Stock Price": [],
        "Second Moment of Stock Price": [],
        "Computational Time (s)": []
    }

    # Black-Scholes
    with contextlib.redirect_stdout(io.StringIO()):
        bs_price, bs_abs_error, bs_mean, bs_second_moment, bs_time = black_scholes(S_0, r, sigma, T, K, call_option)
    results["Model"].append("Black-Scholes")
    results["Option Price"].append(bs_price)
    results["Absolute Error"].append(bs_abs_error)
    results["Mean Stock Price"].append(bs_mean)
    results["Second Moment of Stock Price"].append(bs_second_moment)
    results["Computational Time (s)"].append(bs_time)

    # Binomial CRR
    with contextlib.redirect_stdout(io.StringIO()):
        crr_price, crr_error, crr_mean, crr_second_moment, crr_time = binomial_CRR_optimised_with_error(S_0, r, sigma, T, K, M, call_option)
    results["Model"].append("Approximate Binomial CRR")
    results["Option Price"].append(crr_price)
    results["Absolute Error"].append(crr_error)
    results["Mean Stock Price"].append(crr_mean)
    results["Second Moment of Stock Price"].append(crr_second_moment)
    results["Computational Time (s)"].append(crr_time)

    # Exact CRR
    with contextlib.redirect_stdout(io.StringIO()):
        exact_crr_price, exact_error, exact_mean, exact_second_moment, exact_time = binomial_CRR_exact(S_0, r, sigma, T, K, M, call_option)
    results["Model"].append("Exact Binomial CRR")
    results["Option Price"].append(exact_crr_price)
    results["Absolute Error"].append(exact_error)
    results["Mean Stock Price"].append(exact_mean)
    results["Second Moment of Stock Price"].append(exact_second_moment)
    results["Computational Time (s)"].append(exact_time)

    # Optimal u (from optimal_u)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, optimal_u_price, optimal_error, optimal_mean, optimal_second_moment, optimal_time, _ = optimal_u(S_0, r, sigma, T, K, M, call_option)
    results["Model"].append("Optimal u")
    results["Option Price"].append(optimal_u_price)
    results["Absolute Error"].append(optimal_error)
    results["Mean Stock Price"].append(optimal_mean)
    results["Second Moment of Stock Price"].append(optimal_second_moment)
    results["Computational Time (s)"].append(optimal_time)

    # Tilted Tree
    with contextlib.redirect_stdout(io.StringIO()):
        tilted_price, tilted_error, tilted_mean, tilted_second_moment, tilted_time = tilted_tree(S_0, r, sigma, T, K, M, call_option)
    results["Model"].append("Tilted Tree")
    results["Option Price"].append(tilted_price)
    results["Absolute Error"].append(tilted_error)
    results["Mean Stock Price"].append(tilted_mean)
    results["Second Moment of Stock Price"].append(tilted_second_moment)
    results["Computational Time (s)"].append(tilted_time)

    # Tilted Tree with Richardson
    with contextlib.redirect_stdout(io.StringIO()):
        richardson_price, richardson_error, richardson_mean, richardson_second_moment, richardson_time = tilted_tree_with_richardson(S_0, r, sigma, T, K, M, call_option)
    results["Model"].append("Tilted Tree with Richardson Extrapolation")
    results["Option Price"].append(richardson_price)
    results["Absolute Error"].append(richardson_error)
    results["Mean Stock Price"].append(richardson_mean)
    results["Second Moment of Stock Price"].append(richardson_second_moment)
    results["Computational Time (s)"].append(richardson_time)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Create a unique filename using the M value
    option_type = "call" if call_option else "put"
    unique_filename = f"{base_filename}_{option_type}_M{M}.tex"
    
    # Save the table to a .tex file for use in LaTeX
    with open(unique_filename, "w") as f:
        f.write(df_results.to_latex(index=False, float_format="%.4f"))

    print(f"Table saved as {unique_filename}")
    return df_results

df_48c = generate_comparison_table(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=48, call_option=True)
df_50c = generate_comparison_table(S_0=285, r=0.021, sigma=0.12, T=20, K=290, M=50, call_option=True)

# Part 6
# Convergence plots for exact binomial CRR, tilted trees and Richardson extrapolation
plot_convergence_to_black_scholes(S_0=285, r=0.021, sigma=0.12, T=20, K=290, pricing_function=binomial_CRR_exact, filename = 'exact_convergence.pdf')
plot_convergence_to_black_scholes(S_0=285, r=0.021, sigma=0.12, T=20, K=290, pricing_function=tilted_tree, filename = 'tt_convergence.pdf')
plot_convergence_to_black_scholes(S_0=285, r=0.021, sigma=0.12, T=20, K=290, pricing_function=tilted_tree_with_richardson, filename = 'RE_convergence.pdf')

