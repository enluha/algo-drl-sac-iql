import pandas as pd
import numpy as np

# Load data
trades = pd.read_csv('evaluation/reports/trades_BTCUSDT.csv', index_col='timestamp', parse_dates=True)
equity_df = pd.read_csv('evaluation/reports/equity_curve_BTCUSDT.csv', index_col='timestamp', parse_dates=True)

print("=== EQUITY CALCULATION VERIFICATION ===\n")

# Check first 10 trades
print("First 10 trades:")
print(trades[['raw', 'cost', 'net', 'weight', 'equity']].head(10))

print("\n=== MANUAL EQUITY RECONSTRUCTION ===")
eq_manual = [1.0]
for i in range(min(10, len(trades))):
    eq_manual.append(eq_manual[-1] * (1 + trades['net'].iloc[i]))

print("Manual (compounding net):", [f'{e:.6f}' for e in eq_manual[:11]])
print("Actual (from env):       ", [f"{1.0:.6f}"] + [f"{e:.6f}" for e in trades['equity'].head(10).tolist()])

print("\n=== EQUITY DISCREPANCY CHECK ===")
# Check if equity column matches net returns compounded
equity_from_net = (1 + trades['net']).cumprod()
print(f"Equity from net cumprod: {equity_from_net.iloc[0]:.6f} -> {equity_from_net.iloc[-1]:.6f}")
print(f"Equity from env column:  {trades['equity'].iloc[0]:.6f} -> {trades['equity'].iloc[-1]:.6f}")
print(f"Match? {np.allclose(equity_from_net, trades['equity'])}")

print("\n=== POTENTIAL ISSUES ===")

# Issue 1: Check if equity column is being used directly instead of recomputed
print("1. Equity column consistency:")
reconstructed = (1 + equity_df['equity'].pct_change().fillna(0)).cumprod()
print(f"   Original equity: {equity_df['equity'].iloc[-1]:.6f}")
print(f"   Reconstructed from returns: {reconstructed.iloc[-1]:.6f}")

# Issue 2: Check raw returns vs net returns
print("\n2. Return decomposition:")
print(f"   Sum of raw returns: {trades['raw'].sum():.6f}")
print(f"   Sum of costs: {trades['cost'].sum():.6f}")
print(f"   Sum of net returns: {trades['net'].sum():.6f}")
print(f"   Net should be raw - cost*kappa_cost")

# Issue 3: Check if raw returns are being multiplied by leverage
print("\n3. Leverage analysis:")
print(f"   Average absolute weight: {trades['weight'].abs().mean():.4f}")
print(f"   Max weight: {trades['weight'].abs().max():.4f}")
print(f"   Config leverage_max: 1.0")

# Issue 4: Check the equity calculation formula
print("\n4. Environment equity update formula:")
print("   From code: eq = equity * (1.0 + raw - cost * kappa_cost)")
print("   But raw = weight * log_return")
print("   This compounds BOTH the raw return AND the cost penalty")

print("\n5. Cost penalty multiplier:")
print(f"   kappa_cost from config: {0.00015}")
print(f"   Actual cost = bps * turnover = 0.0015 * turnover")
print(f"   Applied cost = cost * kappa_cost = cost * 0.00015")
print(f"   This makes costs very small!")

# Issue 6: Calculate what sharpe should be
print("\n6. Reality check:")
hours_per_year = 24 * 365
mean_return = trades['net'].mean()
std_return = trades['net'].std()
sharpe = mean_return / std_return * np.sqrt(hours_per_year)
print(f"   Sharpe ratio: {sharpe:.2f}")
print(f"   For reference, Sharpe > 3 is exceptional")
print(f"   Sharpe > 10 is nearly impossible in real trading")
print(f"   Sharpe = 19.4 suggests unrealistic returns or data leakage")

print("\n7. Annualized return calculation:")
total_hours = len(trades)
total_days = total_hours / 24
final_equity = trades['equity'].iloc[-1]
ann_return = (final_equity ** (365 / total_days) - 1) * 100
print(f"   Total hours: {total_hours}")
print(f"   Total days: {total_days:.1f}")
print(f"   Annualized return: {ann_return:.1f}%")
print(f"   (Even 100% annualized is exceptional)")
