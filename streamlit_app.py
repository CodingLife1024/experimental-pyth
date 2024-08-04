import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Black-Scholes functions
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def black_scholes_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    return delta, gamma, theta, vega, rho

def monte_carlo_simulation(S0, K, T, r, sigma, num_simulations=10000):
    S = np.zeros(num_simulations)
    for i in range(num_simulations):
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.normal())
        S[i] = ST
    call_payoffs = np.maximum(S - K, 0)
    put_payoffs = np.maximum(K - S, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)
    return call_price, put_price

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    def objective_function(sigma):
        if option_type == 'call':
            return (black_scholes_call(S, K, T, r, sigma) - option_price) ** 2
        else:
            return (black_scholes_put(S, K, T, r, sigma) - option_price) ** 2

    result = minimize(objective_function, x0=0.2, bounds=[(0.01, 2)])
    return result.x[0]

st.title("🎈 Black-Scholes Option Pricing and Monte Carlo Simulation")

st.sidebar.header("Input Parameters")
current_asset_price = st.sidebar.number_input("Current Asset Price (S)", value=100.06)
strike_price = st.sidebar.number_input("Strike Price (K)", value=100.00)
time_to_maturity = st.sidebar.number_input("Time to Maturity (T) in years", value=1.00)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.20)
risk_free_rate = st.sidebar.number_input("Risk-free Interest Rate (r)", value=0.05)

# Calculate Black-Scholes Prices
call_value = black_scholes_call(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)
put_value = black_scholes_put(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)

st.subheader("Black-Scholes Prices")
st.write(f"Call Price: {call_value:.2f}")
st.write(f"Put Price: {put_value:.2f}")

# Generate a range of underlying asset prices and strike prices
min_spot_price = st.sidebar.number_input("Min Spot Price", value=80.05)
max_spot_price = st.sidebar.number_input("Max Spot Price", value=120.07)
min_volatility = st.sidebar.number_input("Min Volatility for Heatmap", value=0.01)
max_volatility = st.sidebar.number_input("Max Volatility for Heatmap", value=1.00)

S_values = np.linspace(min_spot_price, max_spot_price, 100)
K_values = np.linspace(min_spot_price, max_spot_price, 100)

# Initialize matrices to hold call and put prices
call_prices = np.zeros((len(S_values), len(K_values)))
put_prices = np.zeros((len(S_values), len(K_values)))

# Calculate prices
for i, S in enumerate(S_values):
    for j, K in enumerate(K_values):
        call_prices[i, j] = black_scholes_call(S, K, time_to_maturity, risk_free_rate, volatility)
        put_prices[i, j] = black_scholes_put(S, K, time_to_maturity, risk_free_rate, volatility)

# Display heatmaps for call and put prices
st.subheader("Option Prices Heatmap")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
c1 = ax[0].pcolormesh(K_values, S_values, call_prices, cmap='YlGnBu', shading='auto')
fig.colorbar(c1, ax=ax[0], label='Call Price')
ax[0].set_title('Call Prices Heatmap')
ax[0].set_xlabel('Strike Price (K)')
ax[0].set_ylabel('Underlying Asset Price (S)')

c2 = ax[1].pcolormesh(K_values, S_values, put_prices, cmap='YlOrRd', shading='auto')
fig.colorbar(c2, ax=ax[1], label='Put Price')
ax[1].set_title('Put Prices Heatmap')
ax[1].set_xlabel('Strike Price (K)')
ax[1].set_ylabel('Underlying Asset Price (S)')

st.pyplot(fig)

# Calculate Greeks
delta_matrix = np.zeros((len(S_values), len(K_values)))
gamma_matrix = np.zeros((len(S_values), len(K_values)))
theta_matrix = np.zeros((len(S_values), len(K_values)))
vega_matrix = np.zeros((len(S_values), len(K_values)))
rho_matrix = np.zeros((len(S_values), len(K_values)))

for i, S in enumerate(S_values):
    for j, K in enumerate(K_values):
        delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, time_to_maturity, risk_free_rate, volatility)
        delta_matrix[i, j] = delta
        gamma_matrix[i, j] = gamma
        theta_matrix[i, j] = theta
        vega_matrix[i, j] = vega
        rho_matrix[i, j] = rho

# Display heatmaps for Greeks
st.subheader("Greeks Heatmap")

fig, ax = plt.subplots(2, 3, figsize=(18, 12))

c1 = ax[0, 0].pcolormesh(K_values, S_values, delta_matrix, cmap='coolwarm', shading='auto')
fig.colorbar(c1, ax=ax[0, 0], label='Delta')
ax[0, 0].set_title('Delta Heatmap')
ax[0, 0].set_xlabel('Strike Price (K)')
ax[0, 0].set_ylabel('Underlying Asset Price (S)')

c2 = ax[0, 1].pcolormesh(K_values, S_values, gamma_matrix, cmap='coolwarm', shading='auto')
fig.colorbar(c2, ax=ax[0, 1], label='Gamma')
ax[0, 1].set_title('Gamma Heatmap')
ax[0, 1].set_xlabel('Strike Price (K)')
ax[0, 1].set_ylabel('Underlying Asset Price (S)')

c3 = ax[0, 2].pcolormesh(K_values, S_values, theta_matrix, cmap='coolwarm', shading='auto')
fig.colorbar(c3, ax=ax[0, 2], label='Theta')
ax[0, 2].set_title('Theta Heatmap')
ax[0, 2].set_xlabel('Strike Price (K)')
ax[0, 2].set_ylabel('Underlying Asset Price (S)')

c4 = ax[1, 0].pcolormesh(K_values, S_values, vega_matrix, cmap='coolwarm', shading='auto')
fig.colorbar(c4, ax=ax[1, 0], label='Vega')
ax[1, 0].set_title('Vega Heatmap')
ax[1, 0].set_xlabel('Strike Price (K)')
ax[1, 0].set_ylabel('Underlying Asset Price (S)')

c5 = ax[1, 1].pcolormesh(K_values, S_values, rho_matrix, cmap='coolwarm', shading='auto')
fig.colorbar(c5, ax=ax[1, 1], label='Rho')
ax[1, 1].set_title('Rho Heatmap')
ax[1, 1].set_xlabel('Strike Price (K)')
ax[1, 1].set_ylabel('Underlying Asset Price (S)')

fig.delaxes(ax[1, 2])
st.pyplot(fig)

# Volatility Sensitivity
volatility_range = np.linspace(min_volatility, max_volatility, 10)
call_prices_vol = np.zeros(len(volatility_range))
put_prices_vol = np.zeros(len(volatility_range))

for idx, vol in enumerate(volatility_range):
    call_prices_vol[idx] = black_scholes_call(current_asset_price, strike_price, time_to_maturity, risk_free_rate, vol)
    put_prices_vol[idx] = black_scholes_put(current_asset_price, strike_price, time_to_maturity, risk_free_rate, vol)

st.subheader("Option Prices Sensitivity to Volatility")
fig, ax = plt.subplots()
ax.plot(volatility_range, call_prices_vol, label='Call Price')
ax.plot(volatility_range, put_prices_vol, label='Put Price')
ax.set_title('Option Prices Sensitivity to Volatility')
ax.set_xlabel('Volatility')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Monte Carlo Simulation
num_simulations = st.sidebar.number_input("Number of Simulations", value=10000)

mc_call_price, mc_put_price = monte_carlo_simulation(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility, num_simulations)
bs_call_price = black_scholes_call(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)
bs_put_price = black_scholes_put(current_asset_price, strike_price, time_to_maturity, risk_free_rate, volatility)

st.subheader("Monte Carlo vs Black-Scholes Prices")
st.write(f"Monte Carlo Call Price: {mc_call_price:.2f}")
st.write(f"Black-Scholes Call Price: {bs_call_price:.2f}")
st.write(f"Monte Carlo Put Price: {mc_put_price:.2f}")
st.write(f"Black-Scholes Put Price: {bs_put_price:.2f}")
