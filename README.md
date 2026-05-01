# rtb-sim: Real-Time Bidding Simulation

A DSP-side bidding simulation environment built to demonstrate core adtech concepts: auction mechanics, pCVR-based bid pricing, budget pacing, bid shading, and censored bid landscape estimation.

This is not a toy — each component reflects how real DSP bidding stacks are designed, with explicit attention to the statistical problems that make RTB interesting.

---

## What This Covers

| Component | What it models | Why it's non-trivial |
|---|---|---|
| `AuctionEnvironment` | Second-price and first-price auction mechanics with floor prices | Realistic competitor bid distributions (log-normal), feature-driven true CVR |
| `PCVREstimator` | Logistic regression pCVR model from impression features | Cyclical feature encoding, online buffer retraining, cold-start prior |
| `BudgetPacer` | Throttle-based daily budget pacing | Spend velocity vs. linear pacing curve; participation rate control |
| `BidShader` | First-price bid shading from observed win prices | Adapts shade factor to price dispersion without knowing true competitor bids |
| `BidLandscapeEstimator` | Censored MLE for competitor bid distribution | Wins are observed, losses are right-censored — survival analysis framing |
| `BiddingAgent` | Full DSP agent combining all components | Expected-value bidding: `bid = pCVR × target_CPA` |

---

## Installation

```bash
git clone https://github.com/your-username/rtb-sim.git
cd rtb-sim
pip install -e .
```

---

## Quick Start

```python
from rtb_sim.simulation import run_simulation

df = run_simulation(
    n_impressions=50_000,
    auction_type="second_price",
    target_cpa=50.0,
    daily_budget=5_000.0,
    seed=42,
)

print(df[["won", "clearing_price", "converted", "estimated_pcvr", "true_pcvr"]].head(10))
```

### Compare bidding strategies

```python
from rtb_sim.simulation import compare_strategies

results = compare_strategies(
    n_impressions=30_000,
    target_cpa=50.0,
    daily_budget=3_000.0,
)

import pandas as pd
print(pd.DataFrame(results).T)
```

### Use the landscape estimator directly

```python
from rtb_sim.landscape import BidLandscapeEstimator

lm = BidLandscapeEstimator()

# Feed it wins (observed clearing prices) and losses (our bids that didn't win)
lm.record_win(3.2)
lm.record_loss(2.1)
# ... more observations

# Query win probability at a given bid
print(lm.win_probability(bid=4.0))

# Optimal first-price bid given a true value
print(lm.optimal_bid_first_price(true_value=6.0))
```

---

## Project Structure

```
rtb_sim/
├── rtb_sim/
│   ├── auction.py       # AuctionEnvironment, Impression, AuctionResult
│   ├── agent.py         # BiddingAgent, PCVREstimator, BudgetPacer, BidShader
│   ├── landscape.py     # BidLandscapeEstimator (censored MLE)
│   └── simulation.py    # run_simulation(), compare_strategies()
├── notebooks/
│   └── analysis.py      # Full analysis with plots
├── tests/
│   └── test_core.py     # 15 unit + integration tests
└── requirements.txt
```

---

## Key Design Decisions

### Bid pricing: expected value
```
bid (CPM) = pCVR × target_CPA
```
If an impression has a 2% conversion probability and we're targeting a $50 CPA, the fair bid is $1 CPM. This is the standard DSP bidding framework.

### Bid landscape as a censored estimation problem

When you lose an auction, you don't observe the competitor's bid — you only know it exceeded yours. This is a right-censored observation. The `BidLandscapeEstimator` fits a log-normal model using MLE that correctly accounts for this:

- **Wins** → contribute the full likelihood `f(x | μ, σ)`
- **Losses** → contribute the survival function `P(X > c | μ, σ)` where `c` is our bid

This is directly analogous to survival analysis and is the approach used in production bid landscape models.

### Bid shading (first-price auctions)

In first-price auctions, bidding your true value is suboptimal — you'd win but overpay. The optimal strategy shades the bid downward toward the expected clearing price:

```
shaded_bid = raw_bid × (1 - shade_factor)
```

The shade factor updates from observed win prices: higher variance in clearing prices → more room to shade.

### Budget pacing: throttle-based control

The pacer computes a *pace ratio* (actual spend ÷ expected spend at this point in the day) and adjusts the auction participation rate accordingly:

| Pace ratio | Participation rate |
|---|---|
| > 1.2 (ahead) | 30% |
| 1.05–1.2 (slightly ahead) | 70% |
| 0.8–1.05 (on pace) | 90% |
| < 0.8 (behind) | 100% |

---

## Running Tests

```bash
pytest tests/ -v
```

15 tests covering: auction mechanics, floor prices, pacing logic, shade factor behavior, landscape monotonicity, optimal bid computation, and end-to-end budget adherence.

---

## Running the Full Analysis

```bash
python notebooks/analysis.py
```

Produces:
- `outputs/rtb_analysis.png` — 6-panel visualization
- `outputs/simulation_log.csv` — per-auction log
- `outputs/strategy_comparison.csv` — strategy comparison table
- `outputs/pcvr_calibration.csv` — model calibration by decile

---

## What's Not Here (Extensions Worth Building)

- **Multi-armed bandit for creative/audience selection** — contextual bandit layer on top of the bidding agent
- **Frequency capping** — limit impressions per user per day
- **Deal ID / PMP auctions** — private marketplace mechanics alongside open auction
- **Lookalike modeling** — expand targeting from seed audience using similarity scores
- **Cross-channel budget allocation** — extend pacing across multiple channels simultaneously

---

## Background

Built to demonstrate DSP-side measurement and optimization concepts: auction theory, causal modeling under censoring, and decision-making under budget constraints. The censored likelihood estimation in `landscape.py` and the pacing controller in `agent.py` are the components most directly analogous to production adtech systems.
