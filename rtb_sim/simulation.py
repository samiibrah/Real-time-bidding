"""
simulation.py
-------------
Orchestrates the full auction simulation loop.

Runs N impressions through the environment, passes each to the bidding agent,
records outcomes, and returns a structured auction-level log for analysis.
"""

import numpy as np
import pandas as pd
from typing import Literal

from rtb_sim.auction import AuctionEnvironment
from rtb_sim.agent import BiddingAgent, PCVREstimator
from rtb_sim.landscape import BidLandscapeEstimator


def _simulate_conversion(true_pcvr: float, rng: np.random.Generator) -> bool:
    return bool(rng.random() < true_pcvr)


def warm_start_pcvr_model(
    estimator: PCVREstimator,
    env: AuctionEnvironment,
    n: int = 3000,
    seed: int = 99,
) -> None:
    """Pre-train the pCVR model on synthetic impressions before live bidding."""
    rng = np.random.default_rng(seed)
    imps = [env.sample_impression() for _ in range(n)]
    conversions = [int(rng.random() < imp.true_pcvr) for imp in imps]
    estimator.warm_start(imps, conversions)


def _summarize_df(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "impressions_seen": 0,
            "auctions_entered": 0,
            "wins": 0,
            "win_rate": 0.0,
            "spend": 0.0,
            "conversions": 0,
            "effective_cpa": float("inf"),
        }

    spend = df["clearing_price"].sum() / 1000
    conversions = int(df["converted"].sum())
    return {
        "impressions_seen": int(df["impression_id"].max()),
        "auctions_entered": len(df),
        "wins": int(df["won"].sum()),
        "win_rate": round(float(df["won"].mean()), 4),
        "spend": round(float(spend), 2),
        "conversions": conversions,
        "effective_cpa": round(float(spend / max(conversions, 1)), 2),
    }


def run_simulation(
    n_impressions: int = 50_000,
    auction_type: Literal["first_price", "second_price"] = "second_price",
    target_cpa: float = 50.0,
    daily_budget: float = 5_000.0,
    n_competitors: int = 8,
    competitor_bid_mean: float = 3.5,
    max_bid: float = 20.0,
    bidding_policy: Literal["value", "landscape"] = "value",
    seed: int = 42,
    warm_start: bool = True,
    use_landscape: bool = True,
) -> pd.DataFrame:
    """
    Run a full RTB simulation.

    bidding_policy:
    - "value": pCVR * target CPA * 1000, optionally first-price shaded.
    - "landscape": surplus-maximizing bid using learned P(win) and expected cost.
    """
    rng = np.random.default_rng(seed)

    env = AuctionEnvironment(
        auction_type=auction_type,
        n_competitors=n_competitors,
        competitor_bid_mean=competitor_bid_mean,
        seed=seed,
    )

    landscape = BidLandscapeEstimator(min_obs=50) if use_landscape else None

    agent = BiddingAgent(
        target_cpa=target_cpa,
        daily_budget=daily_budget,
        impressions_per_day=n_impressions,
        auction_type=auction_type,
        max_bid=max_bid,
        bidding_policy=bidding_policy,
        landscape=landscape,
        seed=seed,
    )

    if warm_start:
        warm_start_pcvr_model(agent.pcvr_model, env, n=3000, seed=seed + 1)

    records = []

    for _ in range(n_impressions):
        imp = env.sample_impression()
        bid = agent.bid(imp)

        if bid is None:
            continue

        result = env.run_auction(bid, imp)
        converted = _simulate_conversion(imp.true_pcvr, rng) if result.won else False

        agent.record_result(
            impression=imp,
            bid=bid,
            won=result.won,
            clearing_price=result.clearing_price,
            converted=converted,
        )

        estimated_pcvr = agent.pcvr_model.predict(imp)
        landscape_win_prob = (
            landscape.win_probability(bid)
            if landscape is not None and landscape._fitted
            else np.nan
        )
        expected_clearing_price = (
            landscape.expected_clearing_price(bid)
            if landscape is not None and landscape._fitted
            else np.nan
        )

        records.append({
            "impression_id": imp.impression_id,
            "hour_of_day": imp.hour_of_day,
            "day_of_week": imp.day_of_week,
            "device_type": imp.device_type,
            "vertical": imp.vertical,
            "user_recency_days": round(imp.user_recency_days, 2),
            "true_pcvr": round(imp.true_pcvr, 5),
            "estimated_pcvr": round(estimated_pcvr, 5),
            "estimated_value_cpm": round(estimated_pcvr * target_cpa * 1000, 4),
            "floor_price": round(imp.floor_price, 4),
            "bid": round(bid, 4),
            "won": result.won,
            "clearing_price": round(result.clearing_price, 4),
            "converted": converted,
            "cumulative_spend": round(agent.spend, 2),
            "cumulative_conversions": agent.conversions,
            "cumulative_wins": agent.wins,
            "participation_rate": round(agent.pacer.participation_rate, 3),
            "pace_ratio": round(agent.pacer.pace_ratio, 3),
            "shade_factor": round(agent.shader.current_shade_factor, 3),
            "landscape_fitted": bool(landscape._fitted) if landscape is not None else False,
            "landscape_win_prob": round(float(landscape_win_prob), 4) if not np.isnan(landscape_win_prob) else np.nan,
            "expected_clearing_price": round(float(expected_clearing_price), 4) if not np.isnan(expected_clearing_price) else np.nan,
            "bidding_policy": bidding_policy,
            "auction_type": auction_type,
        })

    return pd.DataFrame(records)


def compare_strategies(
    n_impressions: int = 30_000,
    target_cpa: float = 50.0,
    daily_budget: float = 3_000.0,
    competitor_bid_mean: float = 3.5,
    max_bid: float = 20.0,
    seed: int = 42,
) -> dict:
    """
    Run identical conditions across bidding strategies.

    Strategies:
    1. Flat CPM baseline
    2. Oracle pCVR benchmark
    3. pCVR value bidding, second-price
    4. pCVR value bidding, first-price shaded
    5. pCVR + bid landscape optimization, second-price
    6. pCVR + bid landscape optimization, first-price
    """
    results = {}

    # 1. Flat CPM baseline.
    env = AuctionEnvironment(
        auction_type="second_price",
        competitor_bid_mean=competitor_bid_mean,
        seed=seed,
    )
    agent = BiddingAgent(
        target_cpa=target_cpa,
        daily_budget=daily_budget,
        impressions_per_day=n_impressions,
        auction_type="second_price",
        max_bid=max_bid,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    flat_bid = min(3.0, max_bid)
    for _ in range(n_impressions):
        imp = env.sample_impression()
        result = env.run_auction(flat_bid, imp)
        converted = bool(rng.random() < imp.true_pcvr) if result.won else False
        agent.auctions_entered += 1
        agent.record_result(imp, flat_bid, result.won, result.clearing_price, converted)
    results["flat_cpm_baseline"] = agent.summary()

    # 2. Oracle pCVR benchmark: knows true pCVR, useful as an upper bound.
    env = AuctionEnvironment(
        auction_type="second_price",
        competitor_bid_mean=competitor_bid_mean,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    oracle_spend = 0.0
    oracle_conversions = 0
    oracle_wins = 0
    for _ in range(n_impressions):
        imp = env.sample_impression()
        oracle_bid = min(imp.true_pcvr * target_cpa * 1000, max_bid)
        result = env.run_auction(oracle_bid, imp)
        if result.won:
            oracle_wins += 1
            oracle_spend += result.clearing_price / 1000
            if rng.random() < imp.true_pcvr:
                oracle_conversions += 1
    results["oracle_pcvr"] = {
        "impressions_seen": n_impressions,
        "auctions_entered": n_impressions,
        "wins": oracle_wins,
        "win_rate": round(oracle_wins / n_impressions, 4),
        "spend": round(oracle_spend, 2),
        "conversions": oracle_conversions,
        "effective_cpa": round(oracle_spend / max(oracle_conversions, 1), 2),
    }

    # 3-6. Learned pCVR strategies.
    configs = [
        ("pcvr_second_price", "second_price", "value"),
        ("pcvr_first_price_shaded", "first_price", "value"),
        ("pcvr_landscape_second_price", "second_price", "landscape"),
        ("pcvr_landscape_first_price", "first_price", "landscape"),
    ]

    for label, atype, policy in configs:
        df = run_simulation(
            n_impressions=n_impressions,
            auction_type=atype,
            target_cpa=target_cpa,
            daily_budget=daily_budget,
            competitor_bid_mean=competitor_bid_mean,
            max_bid=max_bid,
            bidding_policy=policy,
            seed=seed,
            use_landscape=True,
        )
        results[label] = _summarize_df(df)

    return results
