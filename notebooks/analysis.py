"""
analysis.py
-----------
Run the RTB simulation and produce analysis outputs.

Execute with: python notebooks/analysis.py
Outputs saved to: outputs/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import lognorm

from rtb_sim.simulation import run_simulation, compare_strategies
from rtb_sim.landscape import BidLandscapeEstimator

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("RTB Simulation — Landscape-Aware Bidding Analysis")
print("=" * 60)

# More realistic defaults than the initial run.
# If you raise competitor_bid_mean, raise max_bid too; otherwise the bidder cannot compete.
COMPETITOR_BID_MEAN = 7.0
TARGET_CPA = 150.0
DAILY_BUDGET = 500.0
MAX_BID = 20.0
N_IMPRESSIONS = 50_000

# ── 1. Run base landscape-optimized simulation ──────────────────
print("\n[1/4] Running landscape-optimized simulation...")

df = run_simulation(
    n_impressions=N_IMPRESSIONS,
    auction_type="second_price",
    target_cpa=TARGET_CPA,
    daily_budget=DAILY_BUDGET,
    competitor_bid_mean=COMPETITOR_BID_MEAN,
    max_bid=MAX_BID,
    bidding_policy="landscape",
    seed=42,
    use_landscape=True,
)

print(f"      Auctions entered: {len(df):,}")
print(f"      Win rate:         {df['won'].mean():.1%}")
print(f"      Conversions:      {df['converted'].sum():,}")
spend = df["clearing_price"].sum() / 1000
convs = df["converted"].sum()
print(f"      Spend:            ${spend:,.2f}")
print(f"      Effective CPA:    ${spend / max(convs, 1):.2f}")
print(f"      Landscape fitted: {df['landscape_fitted'].mean():.1%} of entered auctions")

# ── 2. pCVR calibration ─────────────────────────────────────────
print("\n[2/4] Evaluating pCVR model calibration...")
won_df = df[df["won"]].copy()

if len(won_df) >= 10 and won_df["estimated_pcvr"].nunique() > 1:
    won_df["pcvr_bucket"] = pd.qcut(won_df["estimated_pcvr"], q=10, duplicates="drop")
    calibration = won_df.groupby("pcvr_bucket", observed=True).agg(
        mean_estimated=("estimated_pcvr", "mean"),
        mean_true=("true_pcvr", "mean"),
        n=("true_pcvr", "count"),
    ).reset_index()
else:
    calibration = pd.DataFrame({
        "mean_estimated": [df["estimated_pcvr"].mean()],
        "mean_true": [df["true_pcvr"].mean()],
        "n": [len(df)],
    })

print(calibration[["mean_estimated", "mean_true", "n"]].to_string(index=False))

# ── 3. Bid landscape diagnostics ─────────────────────────────────
print("\n[3/4] Fitting bid landscape model for diagnostics...")
lm = BidLandscapeEstimator(min_obs=50)
wins = df[df["won"] & (df["clearing_price"] > 0)]["clearing_price"].values
losses = df[~df["won"]]["bid"].values
for w in wins:
    lm.record_win(float(w))
for l in losses:
    lm.record_loss(float(l))
print(f"      {lm.summary()}")

# ── 4. Strategy comparison ──────────────────────────────────────
print("\n[4/4] Comparing bidding strategies...")
strategies = compare_strategies(
    n_impressions=30_000,
    target_cpa=TARGET_CPA,
    daily_budget=DAILY_BUDGET,
    competitor_bid_mean=COMPETITOR_BID_MEAN,
    max_bid=MAX_BID,
    seed=42,
)
strat_df = pd.DataFrame(strategies).T
print(strat_df.to_string())

# ── Plots ────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("RTB Simulation — Landscape-Aware Bidding Agent", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")
pct_fmt = FuncFormatter(lambda x, _: f"{x:.1%}")

# Plot 1: Cumulative spend vs conversions over time
ax1 = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()
ax1.plot(df.index, df["cumulative_spend"], linewidth=1.5, label="Spend")
ax1b.plot(df.index, df["cumulative_conversions"], linewidth=1.5, linestyle="--", label="Conversions")
ax1.set_title("Cumulative Spend & Conversions", fontsize=10)
ax1.set_xlabel("Entered auction #")
ax1.set_ylabel("Spend ($)")
ax1b.set_ylabel("Conversions")
ax1.yaxis.set_major_formatter(dollar_fmt)
lines1 = [plt.Line2D([0], [0], lw=1.5), plt.Line2D([0], [0], lw=1.5, linestyle="--")]
ax1.legend(lines1, ["Spend", "Conversions"], fontsize=8)

# Plot 2: pCVR calibration
ax2 = fig.add_subplot(gs[0, 1])
sizes = np.maximum(calibration["n"].values / max(calibration["n"].max(), 1) * 300, 30)
ax2.scatter(calibration["mean_estimated"], calibration["mean_true"], s=sizes, alpha=0.8, zorder=5)
max_pcvr = calibration[["mean_estimated", "mean_true"]].max().max()
lims = [0, max_pcvr * 1.1 if max_pcvr > 0 else 0.1]
ax2.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="Perfect calibration")
ax2.set_title("pCVR Model Calibration", fontsize=10)
ax2.set_xlabel("Estimated pCVR")
ax2.set_ylabel("True CVR")
ax2.legend(fontsize=8)

# Plot 3: Competitor bid distribution vs fitted landscape
ax3 = fig.add_subplot(gs[0, 2])
mu, sigma = lm.params
bid_range = np.linspace(0.1, MAX_BID, 300)
fitted_pdf = lognorm.pdf(bid_range, s=sigma, scale=np.exp(mu))
if len(wins) > 0:
    ax3.hist(wins[wins < MAX_BID], bins=50, density=True, alpha=0.5, label="Observed wins")
ax3.plot(bid_range, fitted_pdf, linewidth=2, label=f"Fitted log-normal\n(μ={mu:.2f}, σ={sigma:.2f})")
ax3.set_title("Bid Landscape: Fitted vs Observed", fontsize=10)
ax3.set_xlabel("Clearing Price (CPM $)")
ax3.set_ylabel("Density")
ax3.legend(fontsize=8)

# Plot 4: Win probability curve
ax4 = fig.add_subplot(gs[1, 0])
bids_test = np.linspace(0.5, MAX_BID, 120)
win_probs = [lm.win_probability(b) for b in bids_test]
ax4.plot(bids_test, win_probs, linewidth=2)
ax4.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax4.set_title("Win Probability vs Bid (CPM)", fontsize=10)
ax4.set_xlabel("Bid (CPM $)")
ax4.set_ylabel("P(win)")
ax4.yaxis.set_major_formatter(pct_fmt)

# Plot 5: Pacing participation rate over time
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df.index, df["participation_rate"], linewidth=1, alpha=0.8)
ax5.set_title("Pacing: Participation Rate Over Time", fontsize=10)
ax5.set_xlabel("Entered auction #")
ax5.set_ylabel("Participation Rate")
ax5.yaxis.set_major_formatter(pct_fmt)
ax5.set_ylim(0, 1.05)

# Plot 6: Strategy comparison bar chart
ax6 = fig.add_subplot(gs[1, 2])
strat_df_plot = strat_df.replace([np.inf, -np.inf], np.nan).copy()
labels = [
    "Flat CPM",
    "Oracle\npCVR",
    "pCVR\n2nd",
    "pCVR\n1st shaded",
    "Landscape\n2nd",
    "Landscape\n1st",
]
strategy_order = [
    "flat_cpm_baseline",
    "oracle_pcvr",
    "pcvr_second_price",
    "pcvr_first_price_shaded",
    "pcvr_landscape_second_price",
    "pcvr_landscape_first_price",
]
cpas = [strat_df_plot.loc[k, "effective_cpa"] if k in strat_df_plot.index else np.nan for k in strategy_order]
bars = ax6.bar(labels, cpas, alpha=0.85, edgecolor="white")
ax6.axhline(TARGET_CPA, linestyle="--", linewidth=1.5, label=f"Target CPA (${TARGET_CPA:.0f})")
ax6.set_title("Strategy Comparison: Effective CPA", fontsize=10)
ax6.set_ylabel("Effective CPA ($)")
ax6.legend(fontsize=8)
ax6.tick_params(axis="x", labelrotation=20)
for bar, val in zip(bars, cpas):
    if not pd.isna(val):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"${val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.savefig("outputs/rtb_analysis.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: outputs/rtb_analysis.png")

# Save data
df.to_csv("outputs/simulation_log.csv", index=False)
strat_df.to_csv("outputs/strategy_comparison.csv")
calibration.to_csv("outputs/pcvr_calibration.csv", index=False)
print("Saved: outputs/simulation_log.csv")
print("Saved: outputs/strategy_comparison.csv")
print("Saved: outputs/pcvr_calibration.csv")
print("\nDone.")
