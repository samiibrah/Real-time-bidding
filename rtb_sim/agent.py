"""
agent.py
--------
DSP-side bidding agent.

Responsibilities:
1. Estimate pCVR per impression from observable features
2. Translate pCVR -> raw bid using expected value pricing
3. Optionally optimize bids against a learned bid landscape
4. Apply bid shading in first-price environments
5. Apply budget pacing throttle
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass, field
from typing import Literal, Optional
import warnings

from rtb_sim.auction import Impression
from rtb_sim.landscape import BidLandscapeEstimator

warnings.filterwarnings("ignore")


@dataclass
class BudgetPacer:
    """
    Throttle-based pacing controller.

    If spend is ahead of a linear daily pacing curve, reduce auction participation.
    If spend is behind, increase participation up to 100%.
    """
    daily_budget: float
    impressions_per_day: int
    _spent: float = field(default=0.0, init=False)
    _impressions_seen: int = field(default=0, init=False)

    @property
    def pace_ratio(self) -> float:
        expected_spend = self.daily_budget * (self._impressions_seen / self.impressions_per_day)
        if expected_spend == 0:
            return 1.0
        return self._spent / (expected_spend + 1e-9)

    @property
    def participation_rate(self) -> float:
        ratio = self.pace_ratio
        if ratio > 1.2:
            return 0.3
        if ratio > 1.05:
            return 0.7
        if ratio < 0.8:
            return 1.0
        return 0.9

    def record_impression(self, spent: float = 0.0) -> None:
        self._impressions_seen += 1
        self._spent += spent

    def reset(self) -> None:
        self._spent = 0.0
        self._impressions_seen = 0

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.daily_budget - self._spent)


class PCVREstimator:
    """
    Logistic pCVR estimator with calibrated probabilities.

    Observable features:
    - hour_of_day and day_of_week using cyclical encoding
    - device_type and vertical using one-hot encoding
    - user_recency_days using log transform
    """

    def __init__(self):
        # LogisticRegression is already reasonably calibrated for this synthetic setup.
        # Keeping this lightweight matters because the model is refit online.
        self.model = LogisticRegression(max_iter=500, C=1.0)
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

    def _featurize_batch(self, impressions: list[Impression]) -> np.ndarray:
        return np.array([self._featurize(imp) for imp in impressions])

    def warm_start(self, impressions: list[Impression], conversions: list[int]) -> None:
        self.device_enc.fit([[d] for d in ["mobile", "desktop", "tablet"]])
        self.vertical_enc.fit([[v] for v in ["e-commerce", "auto", "finance", "telecom"]])
        X = self._featurize_batch(impressions)
        y = np.array(conversions)
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, imp: Impression) -> float:
        if not self._fitted:
            return 0.02
        X = self._featurize(imp).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])

    def update(self, imp: Impression, converted: bool) -> None:
        self._buffer_X.append(imp)
        self._buffer_y.append(int(converted))

        if len(self._buffer_y) >= 1000:
            X = self._featurize_batch(self._buffer_X)
            y = np.array(self._buffer_y)
            if y.sum() > 3:
                self.model.fit(X, y)
            self._buffer_X = []
            self._buffer_y = []


class BidShader:
    """
    Simple first-price bid shading.

    This is intentionally separate from the landscape-optimized policy. The shader is
    useful as a heuristic baseline; the landscape policy is the more DSP-realistic
    next step because it explicitly optimizes surplus against P(win).
    """

    def __init__(self):
        self._win_prices: list[float] = []
        self._loss_bids: list[float] = []
        self._shade_factor: float = 0.15

    def record_win(self, clearing_price: float) -> None:
        if clearing_price > 0:
            self._win_prices.append(clearing_price)
            self._update_shade()

    def record_loss(self, our_bid: float) -> None:
        self._loss_bids.append(our_bid)

    def _update_shade(self) -> None:
        if len(self._win_prices) < 20:
            return
        prices = np.array(self._win_prices[-200:])
        cv = prices.std() / (prices.mean() + 1e-9)
        self._shade_factor = float(np.clip(0.05 + 0.2 * cv, 0.05, 0.35))

    def shade(self, raw_bid: float) -> float:
        return raw_bid * (1 - self._shade_factor)

    @property
    def current_shade_factor(self) -> float:
        return self._shade_factor


class BiddingAgent:
    """
    Full DSP-side bidding agent.

    bidding_policy:
    - "value": bid estimated value directly; optionally shade in first-price auctions.
    - "landscape": choose bid that maximizes expected surplus using a learned bid landscape.
    """

    def __init__(
        self,
        target_cpa: float = 50.0,
        daily_budget: float = 5_000.0,
        impressions_per_day: int = 100_000,
        auction_type: Literal["first_price", "second_price"] = "second_price",
        max_bid: float = 20.0,
        bidding_policy: Literal["value", "landscape"] = "value",
        landscape: Optional[BidLandscapeEstimator] = None,
        min_landscape_obs: int = 250,
        seed: int = 0,
    ):
        self.target_cpa = target_cpa
        self.auction_type = auction_type
        self.max_bid = max_bid
        self.bidding_policy = bidding_policy
        self.landscape = landscape
        self.min_landscape_obs = min_landscape_obs
        self.rng = np.random.default_rng(seed)

        self.pcvr_model = PCVREstimator()
        self.pacer = BudgetPacer(
            daily_budget=daily_budget,
            impressions_per_day=impressions_per_day,
        )
        self.shader = BidShader()

        self.impressions_seen = 0
        self.auctions_entered = 0
        self.wins = 0
        self.spend = 0.0
        self.conversions = 0

    def _estimated_value_cpm(self, impression: Impression) -> float:
        """Expected value in CPM units: pCVR * target CPA * 1000."""
        pcvr = self.pcvr_model.predict(impression)
        return pcvr * self.target_cpa 

    def _landscape_optimized_bid(self, value_cpm: float) -> float:
        """
        Choose the bid that maximizes expected surplus.

        First-price:   P(win|b) * (value - b)
        Second-price:  P(win|b) * (value - E[clearing price | clearing price < b])
        """
        if (
            self.landscape is None
            or not self.landscape._fitted
            or self.landscape.n_observations < self.min_landscape_obs
            or value_cpm <= 0
        ):
            # Fallback while the market model is still learning.
            bid = value_cpm
            if self.auction_type == "first_price":
                bid = self.shader.shade(bid)
            return float(np.clip(bid, 0.01, self.max_bid))

        upper = min(value_cpm, self.max_bid)
        if upper <= 0.01:
            return 0.01

        candidates = np.linspace(0.01, upper, 80)
        win_probs = np.array([self.landscape.win_probability(b) for b in candidates])

        if self.auction_type == "first_price":
            expected_cost = candidates
        else:
            expected_cost = np.array([
                self.landscape.expected_clearing_price(b) for b in candidates
            ])

        expected_surplus = win_probs * (value_cpm - expected_cost)
        best_idx = int(np.argmax(expected_surplus))
        return float(np.clip(candidates[best_idx], 0.01, self.max_bid))

    def bid(self, impression: Impression) -> Optional[float]:
        self.impressions_seen += 1
        self.pacer.record_impression(spent=0.0)

        if self.rng.random() > self.pacer.participation_rate:
            return None

        if self.pacer.remaining_budget <= 0:
            return None

        value_cpm = self._estimated_value_cpm(impression)

        if self.bidding_policy == "landscape":
            bid = self._landscape_optimized_bid(value_cpm)
        else:
            bid = value_cpm
            if self.auction_type == "first_price":
                bid = self.shader.shade(bid)
            bid = float(np.clip(bid, 0.01, self.max_bid))

        self.auctions_entered += 1
        return bid

    def record_result(
        self,
        impression: Impression,
        bid: float,
        won: bool,
        clearing_price: float,
        converted: bool,
    ) -> None:
        """Update trackers, pCVR model, shader, and optional landscape model."""
        if won:
            self.wins += 1
            cost = clearing_price / 1000
            self.spend += cost
            self.pacer._spent += cost
            self.shader.record_win(clearing_price)
            if converted:
                self.conversions += 1
        else:
            self.shader.record_loss(bid)

        if self.landscape is not None:
            if won:
                self.landscape.record_win(clearing_price)
            else:
                self.landscape.record_loss(bid)

        # In RTB, conversion labels are usually only observable for won impressions.
        if won:
            self.pcvr_model.update(impression, converted)

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
