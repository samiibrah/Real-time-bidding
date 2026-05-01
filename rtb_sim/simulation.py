"""
simulation.py
-------------
Orchestrates the full auction simulation loop.

Runs N impressions through the environment, passes each to the bidding agent,
records outcomes, and returns a structured log for analysis.
"""

import numpy as np
import pandas as pd
from typing import Literal

from rtb_sim.auction import AuctionEnvironment
from rtb_sim.agent import BiddingAgent, PCVREstimator
from rtb_sim.landscape import BidLandscapeEstimator


def _simulate_conversion(true_pcvr: float, rng: np.random.Generator) -> bool:
    return bool(rng.random() < true_pcvr)


def warm_start_pcvr_model(estimator: PCVREstimator, env: AuctionEnvironment, n: int = 2000, seed: int = 99):
    """Pre-train the pCVR model on synthetic impressions before live simulation."""
    rng = np.random.default_rng(seed)
    imps = [env.sample_impression() for _ in range(n)]
    conversions = [int(rng.random() < imp.true_pcvr) for imp in imps]
    estimator.warm_start(imps, conversions)


def run_simulation(
    n_impressions: int = 50_000,
    auction_type: Literal["first_price", "second_price"] = "second_price",
    target_cpa: float = 50.0,
    daily_budget: float = 5_000.0,
    n_competitors: int = 8,
    competitor_bid_mean: float = 3.5,
    seed: int = 42,
    warm_start: bool = True,
    use_landscape: bool = True,
) -> pd.DataFrame:
    """
    Run a full RTB simulation.

    Parameters
    ----------
    n_impressions : total impression opportunities to simulate
    auction_type : 'first_price' or 'second_price'
    target_cpa : agent's target cost per acquisition
    daily_budget : agent's total spend cap
    n_competitors : number of competing bidders per auction
    competitor_bid_mean : mean of competitor log-normal bid distribution
    seed : reproducibility seed
    warm_start : pre-train pCVR model before live bidding
    use_landscape : attach and update bid landscape estimator

    Returns
    -------
    pd.DataFrame with one row per auction entered
    """
    rng = np.random.default_rng(seed)

    env = AuctionEnvironment(
        auction_type=auction_type,
        n_competitors=n_competitors,
        competitor_bid_mean=competitor_bid_mean,
        seed=seed,
    )

    agent = BiddingAgent(
        target_cpa=target_cpa,
        daily_budget=daily_budget,
        impressions_per_day=n_impressions,
        auction_type=auction_type,
        seed=seed,
    )

    landscape = BidLandscapeEstimator() if use_landscape else None

    if warm_start:
        warm_start_pcvr_model(agent.pcvr_model, env, n=3000, seed=seed + 1)

    records = []

    for _ in range(n_impressions):
        imp = env.sample_impression()
        bid = agent.bid(imp)

        if bid is None:
            continue  # agent opted out (pacing or budget)

        result = env.run_auction(bid, imp)
        converted = False

        if result.won:
            converted = _simulate_conversion(imp.true_pcvr, rng)

        agent.record_result(imp, result.won, result.clearing_price, converted)

        if landscape:
            if result.won:
                landscape.record_win(result.clearing_price)
            else:
                landscape.record_loss(bid)

        estimated_pcvr = agent.pcvr_model.predict(imp)

        records.append({
            "impression_id": imp.impression_id,
            "hour_of_day": imp.hour_of_day,
            "day_of_week": imp.day_of_week,
            "device_type": imp.device_type,
            "vertical": imp.vertical,
            "user_recency_days": round(imp.user_recency_days, 2),
            "true_pcvr": round(imp.true_pcvr, 5),
            "estimated_pcvr": round(estimated_pcvr, 5),
            "floor_price": round(imp.floor_price, 4),
            "bid": round(bid, 4),
            "won": result.won,
            "clearing_price": round(result.clearing_price, 4),
            "converted": converted,
            "cumulative_spend": round(agent.spend, 2),
            "cumulative_conversions": agent.conversions,
            "cumulative_wins": agent.wins,
            "participation_rate": round(agent.pacer.participation_rate, 3),
            "shade_factor": round(agent.shader.current_shade_factor, 3),
            "landscape_win_prob": round(landscape.win_probability(bid), 4) if landscape and landscape._fitted else None,
        })

    df = pd.DataFrame(records)
    return df


def compare_strategies(
    n_impressions: int = 30_000,
    target_cpa: float = 50.0,
    daily_budget: float = 3_000.0,
    seed: int = 42,
) -> dict:
    """
    Run identical conditions across four bidding strategies and compare.

    Strategies:
    1. Flat CPM bidding (fixed $3 CPM, no pCVR model)
    2. pCVR-based bidding, second-price, no shading
    3. pCVR-based bidding, first-price, with shading
    4. pCVR-based bidding, first-price, landscape-optimized
    """
    results = {}

    # 1. Flat CPM baseline
    env = AuctionEnvironment(auction_type="second_price", seed=seed)
    agent = BiddingAgent(target_cpa=target_cpa, daily_budget=daily_budget,
                         impressions_per_day=n_impressions, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(n_impressions):
        imp = env.sample_impression()
        flat_bid = 3.0  # fixed CPM
        result = env.run_auction(flat_bid, imp)
        converted = bool(rng.random() < imp.true_pcvr) if result.won else False
        agent.record_result(imp, result.won, result.clearing_price, converted)
        agent.auctions_entered += 1
    results["flat_cpm_baseline"] = agent.summary()
        # 1.5 Oracle pCVR strategy (upper bound benchmark)
    env = AuctionEnvironment(auction_type="second_price", seed=seed)
    rng = np.random.default_rng(seed)

    oracle_spend = 0.0
    oracle_conversions = 0
    oracle_wins = 0

    for _ in range(n_impressions):
        imp = env.sample_impression()

        # 👇 THIS is where it goes
        oracle_bid = imp.true_pcvr * target_cpa * 1000

        result = env.run_auction(oracle_bid, imp)

        if result.won:
            oracle_wins += 1
            oracle_spend += result.clearing_price / 1000
            if rng.random() < imp.true_pcvr:
                oracle_conversions += 1

    results["oracle_pcvr"] = {
        "wins": oracle_wins,
        "win_rate": round(oracle_wins / n_impressions, 4),
        "spend": round(oracle_spend, 2),
        "conversions": oracle_conversions,
        "effective_cpa": round(
            oracle_spend / max(oracle_conversions, 1), 2
        ),
    }
    # 2–4. Run full simulations
    for label, atype, shade in [
        ("pcvr_second_price", "second_price", False),
        ("pcvr_first_price_shaded", "first_price", True),
    ]:
        df = run_simulation(
            n_impressions=n_impressions,
            auction_type=atype,
            target_cpa=target_cpa,
            daily_budget=daily_budget,
            seed=seed,
        )
        results[label] = {
            "impressions_seen": len(df),
            "wins": int(df["won"].sum()),
            "win_rate": round(df["won"].mean(), 4),
            "spend": round(df["clearing_price"].sum() / 1000, 2),
            "conversions": int(df["converted"].sum()),
            "effective_cpa": round(
                (df["clearing_price"].sum() / 1000) / max(df["converted"].sum(), 1), 2
            ),
        }

    return results
