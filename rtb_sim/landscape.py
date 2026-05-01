"""
landscape.py
------------
Bid Landscape Estimation via Censored Distribution Fitting.

In real RTB, you observe:
- Wins: you know the clearing price (second-price) or your bid (first-price)
- Losses: you only know your bid was too low — the true max competitor bid is censored

This is a survival analysis / censored MLE problem.

We model competitor bids as log-normal and estimate parameters
using maximum likelihood with censored observations (losses contribute
only the information that the true value exceeded our bid).

This is one of the most technically interesting and realistic components
of a DSP's bidding stack.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BidLandscapeEstimator:
    """
    Fits a log-normal model to competitor bid distribution using
    a mix of:
      - Win observations: clearing prices (second-price) ≈ second-highest bid
      - Loss observations: censored — competitor max > our bid

    Parameters
    ----------
    min_obs : minimum observations before fitting
    """
    min_obs: int = 50

    _win_prices: list = field(default_factory=list)
    _loss_bids: list = field(default_factory=list)
    _mu: float = field(default=np.log(3.5), init=False)
    _sigma: float = field(default=0.6, init=False)
    _fitted: bool = field(default=False, init=False)

    def record_win(self, clearing_price: float):
        """Clearing price in second-price auction ≈ highest competitor bid."""
        if clearing_price > 0:
            self._win_prices.append(clearing_price)
        self._maybe_fit()

    def record_loss(self, our_bid: float):
        """
        We lost — all we know is max competitor bid > our_bid.
        This is a right-censored observation.
        """
        self._loss_bids.append(our_bid)
        self._maybe_fit()

    def _maybe_fit(self):
        n = len(self._win_prices) + len(self._loss_bids)
        if n >= self.min_obs and n % 25 == 0:
            self._fit()

    def _fit(self):
        """
        Censored MLE for log-normal parameters.

        Log-likelihood:
          For uncensored (wins): log f(x | mu, sigma)
          For censored (losses): log (1 - F(c | mu, sigma))
            where c = our bid (we know the true value exceeds c)
        """
        wins = np.array(self._win_prices[-500:])
        losses = np.array(self._loss_bids[-500:])

        def neg_log_likelihood(params):
            mu, log_sigma = params
            sigma = np.exp(log_sigma)
            if sigma <= 0:
                return 1e10

            ll = 0.0
            # Uncensored: observed clearing prices
            if len(wins) > 0:
                log_wins = np.log(wins + 1e-9)
                ll += -0.5 * ((log_wins - mu) / sigma) ** 2 - np.log(sigma) - np.log(wins + 1e-9)
                ll = ll.sum()

            # Censored: losses — P(X > c) = 1 - Phi((log(c) - mu) / sigma)
            if len(losses) > 0:
                from scipy.special import ndtr
                z = (np.log(losses + 1e-9) - mu) / sigma
                survival = 1 - ndtr(z)
                ll += np.log(survival + 1e-12).sum()

            return -ll

        result = minimize(
            neg_log_likelihood,
            x0=[self._mu, np.log(self._sigma)],
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
        )

        if result.success or result.fun < 1e8:
            self._mu = result.x[0]
            self._sigma = float(np.exp(result.x[1]))
            self._fitted = True

    def win_probability(self, bid: float) -> float:
        """
        Estimated probability that our bid wins the auction.
        P(max_competitor < bid) = CDF of fitted log-normal at bid.
        """
        if not self._fitted:
            return 0.5
        return float(lognorm.cdf(bid, s=self._sigma, scale=np.exp(self._mu)))

    def expected_clearing_price(self, bid: float) -> float:
        """
        Expected clearing price in second-price auction given we win.
        E[max_competitor | max_competitor < bid]
        = E[X | X < bid] under fitted log-normal.
        """
        if not self._fitted:
            return bid * 0.85
        # Truncated mean of log-normal below bid
        from scipy.stats import lognorm as ln
        dist = ln(s=self._sigma, scale=np.exp(self._mu))
        cdf_at_bid = dist.cdf(bid)
        if cdf_at_bid < 1e-6:
            return bid * 0.5
        # Numerical integration via percentiles
        p_upper = cdf_at_bid
        quantiles = np.linspace(0.01, 0.99, 200)
        values = dist.ppf(quantiles * p_upper)
        return float(np.mean(values[values < bid]))

    def optimal_bid_second_price(self, true_value: float) -> float:
        """
        In second-price auctions, truth-telling is optimal (bid = value).
        Returns true_value — included for completeness and to verify strategy.
        """
        return true_value

    def optimal_bid_first_price(self, true_value: float, n_points: int = 50) -> float:
        """
        Estimate optimal first-price bid by maximizing expected surplus:
        E[surplus] = (true_value - bid) × P(win at bid)

        Solved numerically over a grid.
        """
        if not self._fitted:
            return true_value * 0.85  # fallback shade

        bids = np.linspace(0.01, true_value, n_points)
        win_probs = np.array([self.win_probability(b) for b in bids])
        surplus = (true_value - bids) * win_probs
        return float(bids[np.argmax(surplus)])

    @property
    def params(self) -> Tuple[float, float]:
        """Return current (mu, sigma) of fitted log-normal."""
        return self._mu, self._sigma

    @property
    def n_observations(self) -> int:
        return len(self._win_prices) + len(self._loss_bids)

    def summary(self) -> dict:
        mu, sigma = self.params
        return {
            "fitted": self._fitted,
            "n_wins": len(self._win_prices),
            "n_losses": len(self._loss_bids),
            "mu": round(mu, 4),
            "sigma": round(sigma, 4),
            "implied_median_bid": round(float(np.exp(mu)), 4),
            "implied_mean_bid": round(float(np.exp(mu + 0.5 * sigma**2)), 4),
        }
