from rtb_sim.simulation import run_simulation
df = run_simulation(n_impressions=50_000, target_cpa=100.0, daily_budget=500.0, seed=42)
print(f"Win rate: {df['won'].mean():.1%}")
print(f"Effective CPA: ${df['clearing_price'].sum()/1000/max(df['converted'].sum(),1):.2f}")