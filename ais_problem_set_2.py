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


class portfolio_features:

    def __init__(self, expected_returns, std_devs, corr_matrix, rf_rate):
        self.expected_returns = expected_returns
        self.std_devs         = std_devs
        self.corr_matrix      = corr_matrix
        self.rf_rate          = rf_rate
        self.excess_returns = self.expected_returns - self.rf_rate
        self.cov_matrix     = np.outer(self.std_devs, self.std_devs) * self.corr_matrix
        self.raw_weights    = np.linalg.inv(self.cov_matrix) @ self.excess_returns

    def weights_optimizer(self):
        """
        Find weights that maximize the Sharpe ratio using the analytical solution:
            w* ∝ Σ⁻¹ (μ - rf)
        then normalize so weights sum to 1.
        """     
        self.weight      = self.raw_weights / self.raw_weights.sum()
        print(f"Weights [Eq, FI, Com]: {[f"{w:.2%}" for w in self.weight]}")
        return self.weight

    def portfolio_expected_return(self):
        """
        Calculate portfolio return.
        """
        self.pf_ex_return = np.dot(self.weight, self.expected_returns)
        print(f"The Portfolio Expected Return is: {self.pf_ex_return:.2%}")
        return 

    def portfolio_volatility(self):
        """
        Calculate portfolio volatility.
        """
        weight_matrix = np.outer(self.weight, self.weight)
        variance      = np.sum(weight_matrix * self.cov_matrix)
        self.pf_volatility = np.sqrt(variance)
        print(f"The Portfolio Expected Volatility is: {self.pf_volatility:.2%}")
        return self.pf_volatility

    def sharpe_ratio(self):
        """
        Calculate portfolio Sharpe Ratio.
        """
        self.pf_sharpe = (self.pf_ex_return - self.rf_rate) / self.pf_volatility
        print(f"Sharpe Ratio: {self.pf_sharpe:.4f}")
        return self.pf_sharpe


    def op_weights(self, risk_aversion=None, target_vol=None, target_ret=None):
        """
        Compute optimal portfolio weights given one of three targeting approaches:
          - risk_aversion : maximizes quadratic utility  w* = (1/lambda) * raw_weights
          - target_vol    : scales tangency portfolio to hit a desired volatility
          - target_ret    : scales tangency portfolio to hit a desired return
        Remainder goes to the risk-free asset.
        """
        if risk_aversion is not None:
            weight_op = (1 / risk_aversion) * self.raw_weights
        elif target_vol is not None:
            alpha     = target_vol / self.pf_volatility
            weight_op = alpha * self.weight
        elif target_ret is not None:
            alpha     = (target_ret - self.rf_rate) / (self.pf_ex_return - self.rf_rate)
            weight_op = alpha * self.weight
        else:
            raise ValueError("Provide one of: risk_aversion, target_vol, target_ret")
        weight_rf_op = 1 - weight_op.sum()
        self.op_portfolio_stats(weight_op, weight_rf_op)

    def op_portfolio_stats(self, weight_op, weight_rf_op):
        """
        Print characteristics of the optimal portfolio.
        """
        all_weights    = list(weight_op) + [weight_rf_op]
        print(f"Weights [Eq, FI, Com, RF]: {[f'{w:.2%}' for w in all_weights]}")

        pf_ex_return_op  = np.dot(weight_op, self.expected_returns) + weight_rf_op * self.rf_rate
        print(f"Expected Return          : {pf_ex_return_op:.2%}")

        weight_matrix    = np.outer(weight_op, weight_op)
        pf_volatility_op = np.sqrt(np.sum(weight_matrix * self.cov_matrix))
        print(f"Volatility               : {pf_volatility_op:.2%}")

        pf_sharpe_op = (pf_ex_return_op - self.rf_rate) / pf_volatility_op
        print(f"Sharpe Ratio             : {pf_sharpe_op:.4f}")


class taa_portfolio:

    def __init__(self, expected_returns, std_devs, corr, strategic_weights):
        self.expected_returns    = expected_returns[:2]
        self.std_devs            = std_devs[:2]
        self.corr                = corr[:2, :2]
        self.strategic_weights   = strategic_weights
        self.cov_matrix          = np.outer(self.std_devs, self.std_devs) * self.corr

    def tactical_weights(self, sigma_te):
        """
        Find weights that maximize expected return subject to tracking error <= sigma_te.
        Active weight: delta = [d, -d], so TE^2 = d^2 * (var_Eq - 2*cov + var_FI).
        Since mu_Eq > mu_FI, set d to its maximum positive value.
        """
        var_Eq  = self.cov_matrix[0, 0]
        cov     = self.cov_matrix[0, 1]
        var_FI  = self.cov_matrix[1, 1]

        d = sigma_te / np.sqrt(var_Eq - 2 * cov + var_FI)

        w_eq = self.strategic_weights[0] + d
        w_fi = self.strategic_weights[1] - d

        tactical = np.array([w_eq, w_fi])
        self.portfolio_stats(tactical)

    def portfolio_stats(self, weights):
        print(f"Weights [Eq, FI]        : {[f'{w:.2%}' for w in weights]}")

        exp_return = np.dot(weights, self.expected_returns)
        print(f"Expected Return         : {exp_return:.2%}")

        weight_matrix = np.outer(weights, weights)
        volatility    = np.sqrt(np.sum(weight_matrix * self.cov_matrix))
        print(f"Volatility              : {volatility:.2%}")

        active_weights = weights - self.strategic_weights
        te_matrix      = np.outer(active_weights, active_weights)
        tracking_error = np.sqrt(np.sum(te_matrix * self.cov_matrix))
        print(f"Tracking Error          : {tracking_error:.2%}")

        sharpe = (exp_return - risk_free_rate) / volatility
        print(f"Sharpe Ratio            : {sharpe:.4f}")

def main():
    pf = portfolio_features(expected_returns, std_devs, corr_matrix, risk_free_rate)

    print("=" * 35)
    print(" Max Sharpe Portfolio")
    print("=" * 35)
    pf.weights_optimizer()
    pf.portfolio_expected_return()
    pf.portfolio_volatility()
    pf.sharpe_ratio()

    print()
    print("=" * 35)
    print(" Optimal Portfolio | Risk Aversion = 8")
    print("=" * 35)
    pf.op_weights(risk_aversion=8)

    print()
    print("=" * 35)
    print(" Optimal Portfolio | Target Return = 7%")
    print("=" * 35)
    pf.op_weights(target_ret=0.07)

    print()
    print("=" * 35)
    print(" Optimal Portfolio | Target Vol = 10%")
    print("=" * 35)
    pf.op_weights(target_vol=0.1)

    print()
    print("=" * 35)
    print(" TAA Portfolio | TE = 3%")
    print("=" * 35)
    taa = taa_portfolio(expected_returns, std_devs, corr_matrix, strategic_weights=np.array([0.60, 0.40]))
    taa.tactical_weights(sigma_te=0.03)

if __name__ == "__main__":
    main()