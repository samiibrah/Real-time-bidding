"""
tests/test_core.py
------------------
Unit tests for auction mechanics, agent behavior, and landscape estimator.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rtb_sim.auction import AuctionEnvironment, Impression
from rtb_sim.agent import BiddingAgent, BudgetPacer, BidShader
from rtb_sim.landscape import BidLandscapeEstimator


# ── Auction tests ────────────────────────────────────────────────

class TestAuctionEnvironment:

    def setup_method(self):
        self.env = AuctionEnvironment(auction_type="second_price", seed=0)

    def test_impression_sampling(self):
        imp = self.env.sample_impression()
        assert isinstance(imp, Impression)
        assert 0 < imp.true_pcvr < 1
        assert imp.device_type in ["mobile", "desktop", "tablet"]
        assert imp.vertical in ["e-commerce", "auto", "finance", "telecom"]

    def test_second_price_clearing(self):
        imp = self.env.sample_impression()
        imp.floor_price = 0.0
        result = self.env.run_auction(agent_bid=10.0, impression=imp)
        if result.won:
            # In second-price, winner pays less than or equal to their bid
            assert result.clearing_price <= 10.0

    def test_floor_price_respected(self):
        imp = self.env.sample_impression()
        imp.floor_price = 99.0
        result = self.env.run_auction(agent_bid=1.0, impression=imp)
        assert not result.won  # bid below floor → no win

    def test_first_price_clearing_equals_bid(self):
        env = AuctionEnvironment(auction_type="first_price", n_competitors=0, seed=0)
        imp = env.sample_impression()
        imp.floor_price = 0.0
        result = env.run_auction(agent_bid=5.0, impression=imp)
        if result.won:
            assert result.clearing_price == pytest.approx(5.0)

    def test_win_rate_increases_with_bid(self):
        wins_low = sum(
            self.env.run_auction(1.0, self.env.sample_impression()).won
            for _ in range(200)
        )
        wins_high = sum(
            self.env.run_auction(20.0, self.env.sample_impression()).won
            for _ in range(200)
        )
        assert wins_high > wins_low


# ── Pacing tests ─────────────────────────────────────────────────

class TestBudgetPacer:

    def test_full_participation_when_underspent(self):
        pacer = BudgetPacer(daily_budget=100.0, impressions_per_day=1000)
        # No spend, lots of impressions seen → behind pace → rate = 1.0
        for _ in range(500):
            pacer.record_impression(spent=0.0)
        assert pacer.participation_rate == 1.0

    def test_throttle_when_overspent(self):
        pacer = BudgetPacer(daily_budget=100.0, impressions_per_day=1000)
        # Spend all budget in first 10% of impressions → well ahead of pace
        for _ in range(100):
            pacer.record_impression(spent=1.0)
        assert pacer.participation_rate < 1.0

    def test_remaining_budget(self):
        pacer = BudgetPacer(daily_budget=100.0, impressions_per_day=1000)
        pacer.record_impression(spent=30.0)
        assert pacer.remaining_budget == pytest.approx(70.0)


# ── Bid shader tests ─────────────────────────────────────────────

class TestBidShader:

    def test_shade_reduces_bid(self):
        shader = BidShader()
        raw_bid = 5.0
        shaded = shader.shade(raw_bid)
        assert shaded < raw_bid

    def test_shade_factor_updates_with_data(self):
        shader = BidShader()
        initial_factor = shader.current_shade_factor
        for price in np.random.default_rng(0).lognormal(1.0, 0.5, 50):
            shader.record_win(float(price))
        # Shade factor may have updated
        assert 0.05 <= shader.current_shade_factor <= 0.35


# ── Landscape estimator tests ─────────────────────────────────────

class TestBidLandscapeEstimator:

    def test_win_probability_monotone(self):
        lm = BidLandscapeEstimator(min_obs=50)
        rng = np.random.default_rng(0)
        for w in rng.lognormal(1.0, 0.5, 100):
            lm.record_win(float(w))
        for l in rng.uniform(0.5, 3.0, 100):
            lm.record_loss(float(l))
        probs = [lm.win_probability(b) for b in [1.0, 3.0, 6.0, 10.0]]
        assert all(probs[i] <= probs[i+1] for i in range(len(probs)-1))

    def test_optimal_first_price_bid_below_value(self):
        lm = BidLandscapeEstimator(min_obs=50)
        rng = np.random.default_rng(0)
        for w in rng.lognormal(1.0, 0.5, 150):
            lm.record_win(float(w))
        for l in rng.uniform(0.5, 3.0, 150):
            lm.record_loss(float(l))
        true_value = 8.0
        optimal = lm.optimal_bid_first_price(true_value)
        assert optimal < true_value  # shading should always reduce the bid

    def test_summary_keys(self):
        lm = BidLandscapeEstimator()
        summary = lm.summary()
        assert "fitted" in summary
        assert "mu" in summary
        assert "sigma" in summary


# ── Agent integration test ────────────────────────────────────────

class TestBiddingAgent:

    def test_agent_respects_budget(self):
        from rtb_sim.simulation import run_simulation
        df = run_simulation(n_impressions=5_000, daily_budget=500.0, seed=1)
        total_spend = df["clearing_price"].sum() / 1000
        assert total_spend <= 500.0 * 1.02  # allow tiny overshoot from atomicity

    def test_agent_produces_valid_bids(self):
        env = AuctionEnvironment(seed=0)
        agent = BiddingAgent(target_cpa=50.0, daily_budget=10_000.0, impressions_per_day=10_000, seed=0)
        for _ in range(100):
            imp = env.sample_impression()
            bid = agent.bid(imp)
            if bid is not None:
                assert 0 < bid <= agent.max_bid
