import numpy as np

# Asset labels
labels = ["Equities", "Fixed Income", "Commodities"]

# Asset data: [Equities, Fixed Income, Commodities]
expected_returns   = np.array([0.10, 0.07, 0.11])
std_devs           = np.array([0.16, 0.09, 0.25])
market_cap_weights = np.array([0.46, 0.49, 0.05])
risk_free_rate     = 0.04

# Next month return: [Equities, Fixed Income, Commodities]
jan_month_returns = np.array([-0.04, 0.03, 0.12])  

# Correlation matrix: [Equities, Fixed Income, Commodities]
corr_matrix = np.array([
    [ 1.00,  0.26,  0.42],
    [ 0.26,  1.00, -0.18],
    [ 0.42, -0.18,  1.00]
])

def portfolio_expected_return(weights, returns):
    """
    Calculate portfolio return.
    """
    print(f"The Portfolio Expected Return is: {np.dot(weights, returns):.2%}")
    return np.dot(weights, returns)

def portfolio_volatility(weights, std_devs, corr_matrix):
    """
    Calculate portfolio volatility.
    """
    weight_matrix = np.outer(weights, weights)
    stdev_matrix  = np.outer(std_devs, std_devs)
    variance      = np.sum(weight_matrix * stdev_matrix * corr_matrix)
    print(f"The Portfolio Expected Volatility is: {np.sqrt(variance):.2%}")
    return np.sqrt(variance)

def sharpe_ratio(e_ret, volatility, rf_rate):
    """
    Calculate portfolio Sharpe Ratio.
    """
    return (e_ret - rf_rate)/volatility

def rebalance(target_weights, monthly_returns):
    """
    Calculate portfolio rebalancing.
    """
    # New weights after price drift
    drifted     = target_weights * (1 + monthly_returns)
    new_weights = drifted / drifted.sum()

    # Rebalancing trades: poesitive = buy, negative = sell
    trades = target_weights - new_weights

    print("--- Drifted Weights vs Target ---")
    for label, nw, tw in zip(labels, new_weights, target_weights):
        print(f"  {label:<15} drifted: {nw:.2%}  target: {tw:.2%}")

    print("\n--- Rebalancing trades (+ buy / - sell) ---")
    for label, t in zip(labels, trades):
        if t == 0:
            continue

        action = "BUY " if t > 0 else "SELL"
        print(f"  {label:<15} {action}: {abs(t):.2%}")

    return new_weights, trades

class portfolios:

    def __init__(self, expected_returns, std_devs, market_cap_weight):
        self.expected_returns = expected_returns
        self.std_devs = std_devs
        self.market_cap_weight = market_cap_weight
    
    def passive_portfolio(self):
        self.weight = self.market_cap_weight
        print(f"Weights [Eq, FI, Com]: {[f'{w:.2%}' for w in self.weight]}")
        self.exp_return  = portfolio_expected_return(self.weight, self.expected_returns)
        self.volatility  = portfolio_volatility(self.weight, self.std_devs, corr_matrix)
        self.sharpe      = sharpe_ratio(self.exp_return, self.volatility, risk_free_rate)
        print(f"Sharpe Ratio: {self.sharpe:.4f}")

    def sixty_forty(self):
        self.weight     = np.array([0.60, 0.40, 0.00])
        print(f"Weights [Eq, FI, Com]: {[f'{w:.2%}' for w in self.weight]}")
        self.exp_return = portfolio_expected_return(self.weight, self.expected_returns)
        self.volatility = portfolio_volatility(self.weight, self.std_devs, corr_matrix)
        self.sharpe     = sharpe_ratio(self.exp_return, self.volatility, risk_free_rate)
        print(f"Sharpe Ratio: {self.sharpe:.4f}")

    def risk_parity(self):
        inv_vol         = 1 / self.std_devs
        self.weight     = inv_vol / inv_vol.sum()
        print(f"Weights [Eq, FI, Com]: {[f'{w:.2%}' for w in self.weight]}")
        self.exp_return = portfolio_expected_return(self.weight, self.expected_returns)
        self.volatility = portfolio_volatility(self.weight, self.std_devs, corr_matrix)
        self.sharpe     = sharpe_ratio(self.exp_return, self.volatility, risk_free_rate)
        print(f"Sharpe Ratio: {self.sharpe:.4f}")


def main():
    portfolio = portfolios(expected_returns, std_devs, market_cap_weights)
    
    print("=" * 20)
    print(" Passive Portfolio")
    print("=" * 20)
    portfolio.passive_portfolio()
    passive_sharpe  = portfolio.sharpe
    print("\n")

    print("=" * 20)
    print(" 60/40 Portfolio")
    print("=" * 20)
    portfolio.sixty_forty()
    sixty_forty_sharpe  = portfolio.sharpe
    sixty_forty_weights = portfolio.weight
    print("\n")

    print("=" * 20)
    print(" Risk Parity")
    print("=" * 20)
    portfolio.risk_parity()
    risk_parity_sharpe  = portfolio.sharpe
    risk_parity_weights = portfolio.weight
    print("\n")

    sharpes = {
        "Passive":     passive_sharpe,
        "60/40":       sixty_forty_sharpe,
        "Risk Parity": risk_parity_sharpe,
    }

    best = max(sharpes, key=sharpes.get)
    print("=" * 20)
    print(f" Best Sharpe: {best} ({sharpes[best]:.4f})")
    print("=" * 20)
    print("\n")

    print("=" * 20)
    print(" Rebalancing")
    print("=" * 20)
    print("\n60/40 Portfolio:")
    rebalance(sixty_forty_weights, jan_month_returns)
    print("\nRisk Parity:")
    rebalance(risk_parity_weights, jan_month_returns)

if __name__ == "__main__":
    main()