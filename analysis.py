"""
analysis.py
-----------
Run the simulation and produce analysis outputs.

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

from rtb_sim.simulation import run_simulation, compare_strategies
from rtb_sim.landscape import BidLandscapeEstimator

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("RTB Simulation — Analysis")
print("=" * 60)

# ── 1. Run base simulation ──────────────────────────────────────
print("\n[1/4] Running base simulation (50k impressions, second-price)...")
df = run_simulation(
    n_impressions=50_000,
    auction_type="second_price",
    target_cpa=50.0,
    daily_budget=5_000.0,
    seed=42,
)
print(f"      Auctions entered: {len(df):,}")
print(f"      Win rate:         {df['won'].mean():.1%}")
print(f"      Conversions:      {df['converted'].sum():,}")
spend = df['clearing_price'].sum() / 1000
convs = df['converted'].sum()
print(f"      Spend:            ${spend:,.2f}")
print(f"      Effective CPA:    ${spend/max(convs,1):.2f}")

# ── 2. pCVR calibration ─────────────────────────────────────────
print("\n[2/4] Evaluating pCVR model calibration...")
won_df = df[df["won"]].copy()
won_df["pcvr_bucket"] = pd.qcut(won_df["estimated_pcvr"], q=10, duplicates="drop")
calibration = won_df.groupby("pcvr_bucket", observed=True).agg(
    mean_estimated=("estimated_pcvr", "mean"),
    mean_true=("true_pcvr", "mean"),
    n=("true_pcvr", "count"),
).reset_index()
print(calibration[["mean_estimated", "mean_true", "n"]].to_string(index=False))

# ── 3. Bid landscape ────────────────────────────────────────────
print("\n[3/4] Fitting bid landscape model...")
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
strategies = compare_strategies(n_impressions=30_000, target_cpa=50.0, daily_budget=3_000.0)
strat_df = pd.DataFrame(strategies).T
print(strat_df.to_string())

# ── Plots ────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("RTB Simulation — Bidding Agent Analysis", fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")
pct_fmt = FuncFormatter(lambda x, _: f"{x:.1%}")

# Plot 1: Cumulative spend vs conversions over time
ax1 = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()
ax1.plot(df.index, df["cumulative_spend"], color="#2563eb", linewidth=1.5, label="Spend")
ax1b.plot(df.index, df["cumulative_conversions"], color="#16a34a", linewidth=1.5, linestyle="--", label="Conversions")
ax1.set_title("Cumulative Spend & Conversions", fontsize=10)
ax1.set_xlabel("Auction #")
ax1.set_ylabel("Spend ($)", color="#2563eb")
ax1b.set_ylabel("Conversions", color="#16a34a")
ax1.yaxis.set_major_formatter(dollar_fmt)
lines1 = [plt.Line2D([0], [0], color="#2563eb", lw=1.5),
          plt.Line2D([0], [0], color="#16a34a", lw=1.5, linestyle="--")]
ax1.legend(lines1, ["Spend", "Conversions"], fontsize=8)

# Plot 2: pCVR calibration
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(calibration["mean_estimated"], calibration["mean_true"],
            s=calibration["n"] / 20, color="#7c3aed", alpha=0.8, zorder=5)
lims = [0, calibration[["mean_estimated", "mean_true"]].max().max() * 1.1]
ax2.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="Perfect calibration")
ax2.set_title("pCVR Model Calibration", fontsize=10)
ax2.set_xlabel("Estimated pCVR")
ax2.set_ylabel("Actual CVR")
ax2.legend(fontsize=8)

# Plot 3: Competitor bid distribution vs fitted
ax3 = fig.add_subplot(gs[0, 2])
from scipy.stats import lognorm
mu, sigma = lm.params
bid_range = np.linspace(0.1, 15, 200)
fitted_pdf = lognorm.pdf(bid_range, s=sigma, scale=np.exp(mu))
ax3.hist(wins[wins < 15], bins=50, density=True, alpha=0.5, color="#f59e0b", label="Observed wins")
ax3.plot(bid_range, fitted_pdf, color="#dc2626", linewidth=2, label=f"Fitted log-normal\n(μ={mu:.2f}, σ={sigma:.2f})")
ax3.set_title("Bid Landscape: Fitted vs Observed", fontsize=10)
ax3.set_xlabel("Clearing Price (CPM $)")
ax3.set_ylabel("Density")
ax3.legend(fontsize=8)

# Plot 4: Win probability curve
ax4 = fig.add_subplot(gs[1, 0])
bids_test = np.linspace(0.5, 12, 100)
win_probs = [lm.win_probability(b) for b in bids_test]
ax4.plot(bids_test, win_probs, color="#0891b2", linewidth=2)
ax4.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax4.set_title("Win Probability vs Bid (CPM)", fontsize=10)
ax4.set_xlabel("Bid (CPM $)")
ax4.set_ylabel("P(win)")
ax4.yaxis.set_major_formatter(pct_fmt)

# Plot 5: Pacing — participation rate over time
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df.index, df["participation_rate"], color="#db2777", linewidth=1, alpha=0.8)
ax5.set_title("Pacing: Participation Rate Over Time", fontsize=10)
ax5.set_xlabel("Auction #")
ax5.set_ylabel("Participation Rate")
ax5.yaxis.set_major_formatter(pct_fmt)
ax5.set_ylim(0, 1.05)

# Plot 6: Strategy comparison bar chart
ax6 = fig.add_subplot(gs[1, 2])
strat_labels = list(strategies.keys())
short_labels = ["Flat CPM\n(baseline)", "pCVR\n2nd price", "pCVR\n1st price\nshaded"]
cpas = [strategies[k].get("effective_cpa", 0) for k in strat_labels]
colors = ["#94a3b8", "#3b82f6", "#8b5cf6"]
bars = ax6.bar(short_labels[:len(cpas)], cpas[:len(short_labels)], color=colors[:len(cpas)], alpha=0.85, edgecolor="white")
ax6.axhline(50, color="#dc2626", linestyle="--", linewidth=1.5, label="Target CPA ($50)")
ax6.set_title("Strategy Comparison: Effective CPA", fontsize=10)
ax6.set_ylabel("Effective CPA ($)")
ax6.legend(fontsize=8)
for bar, val in zip(bars, cpas):
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
print("\nDone.")
