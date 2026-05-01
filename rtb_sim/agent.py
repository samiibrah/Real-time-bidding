"""
agent.py
--------
DSP-side bidding agent.

Responsibilities:
1. Estimate pCVR per impression from observable features
2. Translate pCVR → raw bid using expected value pricing
3. Apply bid shading (first-price environments)
4. Apply budget pacing throttle
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass, field
from typing import Optional
from sklearn.calibration import CalibratedClassifierCV
import warnings

from rtb_sim.auction import Impression

warnings.filterwarnings("ignore")


@dataclass
class BudgetPacer:
    """
    Pacing controller: smooths spend over a daily budget.

    Strategy: throttle-based pacing.
    If spend is ahead of linear pacing curve → reduce bid participation rate.
    If spend is behind → increase participation rate (up to 1.0).

    Parameters
    ----------
    daily_budget : total CPM spend budget for the day (in dollars, normalized to impressions)
    impressions_per_day : expected total impression opportunities per day
    """
    daily_budget: float
    impressions_per_day: int
    _spent: float = field(default=0.0, init=False)
    _impressions_seen: int = field(default=0, init=False)

    @property
    def pace_ratio(self) -> float:
        """Ratio of actual spend to expected spend at this point in the day."""
        expected_spend = self.daily_budget * (self._impressions_seen / self.impressions_per_day)
        if expected_spend == 0:
            return 1.0
        return self._spent / (expected_spend + 1e-9)

    @property
    def participation_rate(self) -> float:
        """
        Throttle: probability of entering a given auction.
        Ahead of pace → throttle down. Behind pace → open up.
        """
        ratio = self.pace_ratio
        if ratio > 1.2:
            return 0.3   # significantly ahead — throttle hard
        elif ratio > 1.05:
            return 0.7   # slightly ahead — soft throttle
        elif ratio < 0.8:
            return 1.0   # behind — participate in everything
        else:
            return 0.9   # on pace

    def record_impression(self, spent: float = 0.0):
        self._impressions_seen += 1
        self._spent += spent

    def reset(self):
        self._spent = 0.0
        self._impressions_seen = 0

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.daily_budget - self._spent)


class PCVREstimator:
    """
    Logistic regression model to estimate conversion probability
    from observable impression features.

    Features used (observable by DSP):
    - hour_of_day (cyclical encoding)
    - day_of_week
    - device_type (OHE)
    - vertical (OHE)
    - user_recency_days (log-transformed)
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=500, C=1.0)
        base_model = LogisticRegression(max_iter=500, C=1.0)
        self.model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",   # use "sigmoid" for Platt scaling
        cv=3
)
        self.device_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.vertical_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._fitted = False
        self._buffer_X = []
        self._buffer_y = []

    def _featurize(self, imp: Impression) -> np.ndarray:
        hour_sin = np.sin(2 * np.pi * imp.hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * imp.hour_of_day / 24)
        dow_sin = np.sin(2 * np.pi * imp.day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * imp.day_of_week / 7)
        recency = np.log1p(imp.user_recency_days)
        device_ohe = self.device_enc.transform([[imp.device_type]])[0]
        vertical_ohe = self.vertical_enc.transform([[imp.vertical]])[0]
        return np.concatenate([
            [hour_sin, hour_cos, dow_sin, dow_cos, recency],
            device_ohe,
            vertical_ohe,
        ])

    def _featurize_batch(self, impressions: list) -> np.ndarray:
        rows = []
        for imp in impressions:
            hour_sin = np.sin(2 * np.pi * imp.hour_of_day / 24)
            hour_cos = np.cos(2 * np.pi * imp.hour_of_day / 24)
            dow_sin = np.sin(2 * np.pi * imp.day_of_week / 7)
            dow_cos = np.cos(2 * np.pi * imp.day_of_week / 7)
            recency = np.log1p(imp.user_recency_days)
            device_ohe = self.device_enc.transform([[imp.device_type]])[0]
            vertical_ohe = self.vertical_enc.transform([[imp.vertical]])[0]
            rows.append(np.concatenate([
                [hour_sin, hour_cos, dow_sin, dow_cos, recency],
                device_ohe,
                vertical_ohe,
            ]))
        return np.array(rows)

    def warm_start(self, impressions: list, conversions: list):
        """Pre-train on synthetic data before live bidding."""
        self.device_enc.fit([[d] for d in ["mobile", "desktop", "tablet"]])
        self.vertical_enc.fit([[v] for v in ["e-commerce", "auto", "finance", "telecom"]])
        X = self._featurize_batch(impressions)
        y = np.array(conversions)
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, imp: Impression) -> float:
        if not self._fitted:
            return 0.02  # cold start prior
        X = self._featurize(imp).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def update(self, imp: Impression, converted: bool):
        """Online buffer update — retrain periodically."""
        self._buffer_X.append(imp)
        self._buffer_y.append(int(converted))
        if len(self._buffer_y) >= 200:
            X = self._featurize_batch(self._buffer_X)
            y = np.array(self._buffer_y)
            if y.sum() > 3:  # need positive examples
                self.model.fit(X, y)
            self._buffer_X = []
            self._buffer_y = []


class BidShader:
    """
    Bid shading for first-price auctions.

    In first-price auctions, winning at your true value is suboptimal —
    you should shade down toward the estimated clearing price.

    Approach: maintain a running estimate of the competitive bid distribution
    (log-normal parameters) from wins and losses (censored observations),
    then shade bid to expected clearing price + small margin.

    This is a simplified version of the censored MLE approach used in practice.
    """

    def __init__(self):
        self._win_prices: list = []    # prices we paid when winning
        self._loss_bids: list = []     # our bids when losing (censored — true max > our bid)
        self._shade_factor: float = 0.15  # default shade

    def record_win(self, clearing_price: float):
        self._win_prices.append(clearing_price)
        self._update_shade()

    def record_loss(self, our_bid: float):
        self._loss_bids.append(our_bid)

    def _update_shade(self):
        """
        Update shade factor from observed win prices.
        As we accumulate data, estimate what fraction below our raw bid
        we could have bid and still won.
        """
        if len(self._win_prices) < 20:
            return
        prices = np.array(self._win_prices[-200:])
        # If our clearing prices are consistently much lower than our bids,
        # we can shade more. Use win price dispersion as proxy.
        cv = prices.std() / (prices.mean() + 1e-9)
        # Higher variance → more room to shade
        self._shade_factor = float(np.clip(0.05 + 0.2 * cv, 0.05, 0.35))

    def shade(self, raw_bid: float) -> float:
        return raw_bid * (1 - self._shade_factor)

    @property
    def current_shade_factor(self) -> float:
        return self._shade_factor


class BiddingAgent:
    """
    Full DSP-side bidding agent.

    Workflow per impression:
    1. Check pacing — should we participate?
    2. Estimate pCVR
    3. Compute raw bid = pCVR × target_CPA (expected value pricing)
    4. Apply bid shading (first-price only)
    5. Submit bid

    Parameters
    ----------
    target_cpa : target cost per acquisition in dollars
    daily_budget : daily spend cap in dollars
    impressions_per_day : expected auction volume per day (for pacing)
    auction_type : 'first_price' or 'second_price'
    max_bid : hard bid ceiling (CPM)
    """

    def __init__(
        self,
        target_cpa: float = 50.0,
        daily_budget: float = 10_000.0,
        impressions_per_day: int = 100_000,
        auction_type: str = "second_price",
        max_bid: float = 25.0,
        seed: int = 0,
    ):
        self.target_cpa = target_cpa
        self.auction_type = auction_type
        self.max_bid = max_bid
        self.rng = np.random.default_rng(seed)

        self.pcvr_model = PCVREstimator()
        self.pacer = BudgetPacer(
            daily_budget=daily_budget,
            impressions_per_day=impressions_per_day,
        )
        self.shader = BidShader()

        # Tracking
        self.impressions_seen = 0
        self.auctions_entered = 0
        self.wins = 0
        self.spend = 0.0
        self.conversions = 0

    def bid(self, impression: Impression) -> Optional[float]:
        """
        Returns bid (CPM) or None if agent opts out of this auction.
        """
        self.impressions_seen += 1
        self.pacer.record_impression(spent=0.0)

        # Pacing check — throttle participation
        if self.rng.random() > self.pacer.participation_rate:
            return None

        # Budget exhausted
        if self.pacer.remaining_budget <= 0:
            return None

        # Estimate pCVR
        pcvr = self.pcvr_model.predict(impression)

        # Expected value bid: pCVR × target_CPA
        # CPA is in dollars; CPM bid needs impression-level scaling
        # Assume 1000 impressions per conversion opportunity → CPM = pCVR × CPA
        raw_bid = pcvr * self.target_cpa

        # Apply shading for first-price auctions
        if self.auction_type == "first_price":
            raw_bid = self.shader.shade(raw_bid)

        # Hard ceiling
        bid = float(np.clip(raw_bid, 0.01, self.max_bid))
        self.auctions_entered += 1
        return bid

    def record_result(self, impression: Impression, won: bool, clearing_price: float, converted: bool):
        """Update models and trackers after auction result is observed."""
        if won:
            self.wins += 1
            self.spend += clearing_price / 1000  # CPM → per-impression cost
            self.pacer._spent += clearing_price / 1000
            if not won and self.auction_type == "first_price":
                self.shader.record_loss(bid)
            #if self.auction_type == "first_price":
                #self.shader.record_win(clearing_price)
            if converted:
                self.conversions += 1
        if not won and self.auction_type == "first_price":
            self.shader.record_loss(bid)
        else:
            bid = self.max_bid  # proxy for loss bid (actual bid not stored here)
            

        # Update pCVR model
        if won:
            #self.pcvr_model.update(impression, converted)
            self.pcvr_model.update(impression, converted if won else False)

    @property
    def effective_cpa(self) -> float:
        if self.conversions == 0:
            return float("inf")
        return self.spend / self.conversions

    @property
    def win_rate(self) -> float:
        if self.auctions_entered == 0:
            return 0.0
        return self.wins / self.auctions_entered

    def summary(self) -> dict:
        return {
            "impressions_seen": self.impressions_seen,
            "auctions_entered": self.auctions_entered,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "spend": round(self.spend, 2),
            "conversions": self.conversions,
            "effective_cpa": round(self.effective_cpa, 2),
            "budget_remaining": round(self.pacer.remaining_budget, 2),
            "shade_factor": round(self.shader.current_shade_factor, 3),
        }
