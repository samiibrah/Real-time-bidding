"""
auction.py
----------
Simulates an ad auction environment (both first-price and second-price).

Impression opportunities are generated with contextual features.
N competing bidders drawn from configurable distributions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Impression:
    """A single impression opportunity."""
    impression_id: int
    hour_of_day: int          # 0–23
    day_of_week: int          # 0–6
    device_type: str          # mobile | desktop | tablet
    vertical: str             # e-commerce | auto | finance | telecom
    user_recency_days: float  # days since last site visit (proxy for intent)
    true_pcvr: float          # ground-truth conversion probability (unobserved by agent)
    floor_price: float        # auction floor (CPM)


@dataclass
class AuctionResult:
    impression: Impression
    winner_bid: float
    clearing_price: float  # what winner pays
    won: bool
    auction_type: Literal["first_price", "second_price"]


class AuctionEnvironment:
    """
    Generates impression opportunities and runs auctions.

    Parameters
    ----------
    auction_type : 'first_price' or 'second_price'
    n_competitors : number of competing bidders in each auction
    competitor_bid_mean : mean of log-normal competitor bid distribution (CPM)
    competitor_bid_sigma : sigma of log-normal competitor bid distribution
    seed : random seed for reproducibility
    """

    DEVICE_TYPES = ["mobile", "desktop", "tablet"]
    VERTICALS = ["e-commerce", "auto", "finance", "telecom"]

    # Base CVR by vertical — ground truth the agent never sees directly
    BASE_CVR = {
        "e-commerce": 0.035,
        "auto":        0.018,
        "finance":     0.022,
        "telecom":     0.028,
    }

    def __init__(
        self,
        auction_type: Literal["first_price", "second_price"] = "second_price",
        n_competitors: int = 8,
        competitor_bid_mean: float = 3.5,   # CPM
        competitor_bid_sigma: float = 0.6,
        floor_price_mean: float = 0.5,
        seed: int = 42,
    ):
        self.auction_type = auction_type
        self.n_competitors = n_competitors
        self.competitor_bid_mean = competitor_bid_mean
        self.competitor_bid_sigma = competitor_bid_sigma
        self.floor_price_mean = floor_price_mean
        self.rng = np.random.default_rng(seed)
        self._impression_counter = 0

    def _true_pcvr(self, imp: Impression) -> float:
        """Generate ground-truth CVR with feature effects."""
        base = self.BASE_CVR[imp.vertical]
        # Recency boost: users who visited recently convert more
        recency_effect = np.exp(-imp.user_recency_days / 7) * 0.02
        # Hour-of-day effect: peak intent hours
        hour_effect = 0.005 * np.sin(np.pi * imp.hour_of_day / 12)
        # Device: desktop converts slightly better
        device_effect = 0.005 if imp.device_type == "desktop" else 0.0
        raw = base + recency_effect + hour_effect + device_effect
        return float(np.clip(raw, 0.001, 0.15))

    def sample_impression(self) -> Impression:
        """Sample one impression opportunity."""
        self._impression_counter += 1
        device = self.rng.choice(self.DEVICE_TYPES, p=[0.55, 0.35, 0.10])
        vertical = self.rng.choice(self.VERTICALS)
        hour = int(self.rng.integers(0, 24))
        dow = int(self.rng.integers(0, 7))
        recency = float(self.rng.exponential(scale=5.0))
        floor = float(np.clip(self.rng.normal(self.floor_price_mean, 0.1), 0.1, 2.0))

        imp = Impression(
            impression_id=self._impression_counter,
            hour_of_day=hour,
            day_of_week=dow,
            device_type=device,
            vertical=vertical,
            user_recency_days=recency,
            true_pcvr=0.0,  # filled below
            floor_price=floor,
        )
        imp.true_pcvr = self._true_pcvr(imp)
        return imp

    def _competitor_bids(self) -> np.ndarray:
        """Draw competitor bids from log-normal distribution."""
        return self.rng.lognormal(
            mean=np.log(self.competitor_bid_mean) - 0.5 * self.competitor_bid_sigma ** 2,
            sigma=self.competitor_bid_sigma,
            size=self.n_competitors,
        )

    def run_auction(self, agent_bid: float, impression: Impression) -> AuctionResult:
        """
        Run a single auction given agent's bid (CPM).
        Returns AuctionResult with win/loss and clearing price.
        """
        competitor_bids = self._competitor_bids()
        all_bids = np.append(competitor_bids, agent_bid)
        max_competitor = competitor_bids.max() if len(competitor_bids) > 0 else 0.0

        # Floor price applies
        if agent_bid < impression.floor_price:
            return AuctionResult(
                impression=impression,
                winner_bid=agent_bid,
                clearing_price=0.0,
                won=False,
                auction_type=self.auction_type,
            )

        max_competitor = competitor_bids.max() if len(competitor_bids) > 0 else 0.0
        won = agent_bid > max_competitor and agent_bid >= impression.floor_price

        if self.auction_type == "second_price":
            clearing_price = max(max_competitor, impression.floor_price) if won else 0.0
        else:  # first_price
            clearing_price = agent_bid if won else 0.0

        return AuctionResult(
            impression=impression,
            winner_bid=agent_bid,
            clearing_price=clearing_price,
            won=won,
            auction_type=self.auction_type,
        )
