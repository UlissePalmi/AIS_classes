import os
import sys
import io
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def capture(fn):
    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    fn()
    sys.stdout = old
    return buf.getvalue()

# ── Data loading ──────────────────────────────────────────────────────────────
return_data          = pd.read_excel("data/Project_Data.xlsx", sheet_name="ReturnData",            index_col=0)
momentum_signal      = pd.read_excel("data/Project_Data.xlsx", sheet_name="Momentum Signal",       index_col=0)
other_factor_returns = pd.read_excel("data/Project_Data.xlsx", sheet_name="Other Factor Returns",  index_col=0)

ret = return_data.loc[momentum_signal.index]

# ── Q1 calculations ───────────────────────────────────────────────────────────
ranks        = momentum_signal.rank(axis=1, ascending=False)
high_weights = (ranks <= 8) * (1 / 8)
low_weights  = (ranks >= 9) * (1 / 8)
high_returns = (high_weights * ret).sum(axis=1)
low_returns  = (low_weights  * ret).sum(axis=1)
ls_returns   = high_returns - low_returns

def print_q1():
    print(f"  High return (Jan 1975): {high_returns.iloc[0]:.7f}")
    print(f"  Low  return (Jan 1975): {low_returns.iloc[0]:.7f}")
    print()
    print("  First 12 months of Long-Short returns:")
    first12 = pd.DataFrame({
        "Date":        ls_returns.index[:12],
        "High_Return": high_returns.iloc[:12].values,
        "Low_Return":  low_returns.iloc[:12].values,
        "LS_Return":   ls_returns.iloc[:12].values,
    })
    print(first12.to_string(index=False, float_format=lambda x: f"{x:.7f}"))

# ── Q2 calculations ───────────────────────────────────────────────────────────
ann_mean = ls_returns.mean() * 12
ann_std  = ls_returns.std()  * np.sqrt(12)
sharpe   = ann_mean / ann_std

def print_q2():
    print(f"  Annualized Mean Return : {ann_mean:.4f} ({ann_mean*100:.2f}%)")
    print(f"  Annualized Std Dev     : {ann_std:.4f} ({ann_std*100:.2f}%)")
    print(f"  Sharpe Ratio           : {sharpe:.4f}")

# ── Q3 calculations ───────────────────────────────────────────────────────────
cum_ret        = (1 + ls_returns).cumprod()
hwm            = cum_ret.cummax()
drawdown       = (cum_ret - hwm) / hwm
max_dd         = drawdown.min()
trough_idx     = drawdown.idxmin()
before_trough  = cum_ret.loc[:trough_idx]
peak_val       = hwm.loc[trough_idx]
peak_idx       = before_trough[before_trough == peak_val].index[-1]
post_trough    = cum_ret.loc[cum_ret.index > trough_idx]
recovery_dates = post_trough[post_trough >= peak_val]
recovery_idx   = recovery_dates.index[0] if len(recovery_dates) > 0 else "Never recovered"

def print_q3():
    print(f"  Max Drawdown           : {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  Drawdown Start (Peak)  : {peak_idx}")
    print(f"  Trough                 : {trough_idx}")
    print(f"  Recovery Date          : {recovery_idx}")
    if not isinstance(recovery_idx, str):
        months_to_recover = len(ls_returns.loc[peak_idx:recovery_idx]) - 1
        print(f"  Months to Recover      : {months_to_recover}")

# ── Q4 calculations ───────────────────────────────────────────────────────────
target_vol   = 0.12
scale_factor = target_vol / ann_std
leverage     = 2 * scale_factor

def print_q4():
    print(f"  Current Ann. Volatility: {ann_std*100:.2f}%")
    print(f"  Scale Factor           : {scale_factor:.4f}")
    print(f"  Required Leverage      : {leverage:.4f}x")
    print(f"  (Each side weight      : {scale_factor:.4f})")

# ── Q5 calculations ───────────────────────────────────────────────────────────
mktrf     = other_factor_returns["mktrf"]
model5    = sm.OLS(ls_returns, sm.add_constant(mktrf)).fit()
sigma_eps = np.sqrt(model5.mse_resid)

def print_q5():
    print(model5.summary())
    print(f"  Idiosyncratic Vol (sigma_e) monthly: {sigma_eps:.6f}")
    print(f"  Idiosyncratic Vol (sigma_e) annual : {sigma_eps * np.sqrt(12):.6f}")

# ── Q6 calculations ───────────────────────────────────────────────────────────
alpha5    = model5.params["const"]
mean_mktrf = mktrf.mean()
var_mktrf  = mktrf.var()
var_eps    = model5.mse_resid
w_ls  = (alpha5 / var_eps) / (mean_mktrf / var_mktrf)
w_mkt = 1 - w_ls

def print_q6():
    print(f"  Optimal Weight in LS Strategy : {w_ls:.4f}")
    print(f"  Optimal Weight in Market Port.: {w_mkt:.4f}")
    print(f"  (Inputs: alpha={alpha5:.6f}, sigma_e^2={var_eps:.6f},")
    print(f"   mean_mktrf={mean_mktrf:.6f}, var_mktrf={var_mktrf:.6f})")

# ── Q7 calculations ───────────────────────────────────────────────────────────
factors = ["mktrf", "smb", "hml", "umd"]
corrs   = {f: ls_returns.corr(other_factor_returns[f]) for f in factors}

def print_q7():
    for f, c in corrs.items():
        print(f"  corr(LS, {f.upper():5s}) : {c:.4f}")

# ── Q8 calculations ───────────────────────────────────────────────────────────
ff3       = other_factor_returns[["mktrf", "smb", "hml"]]
c4        = other_factor_returns[["mktrf", "smb", "hml", "umd"]]
model_ff3 = sm.OLS(ls_returns, sm.add_constant(ff3)).fit()
model_c4  = sm.OLS(ls_returns, sm.add_constant(c4)).fit()

def print_q8():
    print("Fama-French 3-Factor Model (LS ~ mktrf + smb + hml)")
    print(model_ff3.summary())
    print()
    print("Carhart 4-Factor Model (LS ~ mktrf + smb + hml + umd)")
    print(model_c4.summary())
    print()
    print("Model Choice: Carhart 4-Factor is used to assess incremental alpha.")
    print("Rationale: UMD captures single-stock momentum; controlling for it isolates")
    print("the alpha that is purely due to industry-level momentum, not stock-level.")

# ── Drawdown Chart ────────────────────────────────────────────────────────────
dates = pd.to_datetime(drawdown.index.astype(str), format="%Y%m")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates, drawdown * 100, color="black", linewidth=1, label="Drawdown")
ax.axhline(max_dd * 100, color="gray", linewidth=1.2, linestyle="--", label="Max Drawdown")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlim(dates[0], dates[-1])
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(loc="lower center", ncol=2, frameon=False)
ax.set_title("Drawdown --- Industry Momentum Long-Short Strategy")
plt.tight_layout()
plt.savefig("latex/drawdown_chart.png", dpi=150)
plt.close()
print("Drawdown chart saved to latex/drawdown_chart.png")

# ── Write results.tex ─────────────────────────────────────────────────────────
sections = [
    ("Q1: Long-Short Portfolio Construction",       capture(print_q1)),
    ("Q2: Annualized Performance",                  capture(print_q2)),
    ("Q3: Maximum Drawdown",                        capture(print_q3)),
    ("Q4: Leverage for 12\\% Volatility Target",    capture(print_q4)),
    ("Q5: Single-Factor Regression",                capture(print_q5)),
    ("Q6: Optimal Two-Asset Portfolio",             capture(print_q6)),
    ("Q7: Factor Correlations",                     capture(print_q7)),
    ("Q8: Multi-Factor Regressions",                capture(print_q8)),
]

tex_path = "latex/results.tex"
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("% Auto-generated by ais_project.py -- do not edit manually\n\n")
    for title, content in sections:
        f.write(f"\\section{{{title}}}\n\n")
        f.write("\\begin{verbatim}\n")
        f.write(content.rstrip())
        f.write("\n\\end{verbatim}\n\n")
    f.write("\\begin{figure}[htbp]\n")
    f.write("    \\centering\n")
    f.write("    \\includegraphics[width=\\linewidth]{drawdown_chart.png}\n")
    f.write("    \\caption{Drawdown --- Industry Momentum Long-Short Strategy}\n")
    f.write("\\end{figure}\n")
print(f"LaTeX results written to {tex_path}")
