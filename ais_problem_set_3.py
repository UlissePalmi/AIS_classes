import pandas as pd

def fix_index(df):
    df = df.copy()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.set_index('Date', inplace=True)
    return df

def compute_turnover(fav_w, unfav_w):
    combined = fav_w - unfav_w
    return (combined.diff().abs().sum(axis=1) / 2).dropna()

def long_short_portfolio(signal_df, returns_df):
    ranks   = signal_df.rank(axis=1, ascending=False)
    fav_w   = ranks.map(lambda r: 1/3 if r <= 3 else 0)
    unfav_w = ranks.map(lambda r: 1/3 if r >= 4 else 0)
    fav_ret   = (fav_w   * returns_df).sum(axis=1)
    unfav_ret = (unfav_w * returns_df).sum(axis=1)
    ls_ret    = fav_ret - unfav_ret
    return fav_w, unfav_w, fav_ret, unfav_ret, ls_ret

### Data Loading

df = pd.read_excel("data/Problem_Set_3_Data.xlsx")
momentum_df = fix_index(df.iloc[:, 0:7])
BM_df       = fix_index(df.iloc[:, [0] + list(range(8, 14))])
returns_df  = fix_index(df.iloc[:, [0] + list(range(15, 21))])


### Question 1 — Momentum portfolio weights

fav_w_mom, unfav_w_mom, fav_ret_mom, unfav_ret_mom, ls_ret_mom = \
    long_short_portfolio(momentum_df, returns_df)


### Question 2 — Momentum portfolio returns

returns_mom = pd.DataFrame({
    "Favorable Portfolio Returns":   fav_ret_mom,
    "Unfavorable Portfolio Returns": unfav_ret_mom,
    "Long-Short Portfolio Returns":  ls_ret_mom,
})


### Question 3 — B/M portfolio weights and turnover comparison

# 3i: B/M weights (same structure as Q1, using B/M signal)
fav_w_bm, unfav_w_bm, fav_ret_bm, unfav_ret_bm, ls_ret_bm = \
    long_short_portfolio(BM_df, returns_df)

returns_bm = pd.DataFrame({
    "Favorable Portfolio Returns":   fav_ret_bm,
    "Unfavorable Portfolio Returns": unfav_ret_bm,
    "Long-Short Portfolio Returns":  ls_ret_bm,
})

# 3ii: Turnover comparison
avg_turnover_mom = compute_turnover(fav_w_mom, unfav_w_mom).mean()
avg_turnover_bm  = compute_turnover(fav_w_bm,  unfav_w_bm).mean()
higher = "Momentum" if avg_turnover_mom > avg_turnover_bm else "B/M"
print(f"{higher} requires more turnover")


### Question 4 — Signal-weighted favorable momentum portfolio (January 2018 only)

jan_signals = momentum_df.loc["2018-1"]
top3_mask   = jan_signals.rank(ascending=False) <= 3
top3_signals = jan_signals.where(top3_mask, 0)
signal_weights_jan = (top3_signals / top3_signals.sum()).to_frame().T
signal_weights_jan.index = ["2018-1"]


### Excel Export

with pd.ExcelWriter("output/PS3_Results.xlsx", engine="openpyxl") as writer:
    fav_w_mom.to_excel(writer,   sheet_name="Favorable_Momentum")
    unfav_w_mom.to_excel(writer, sheet_name="Unfavorable_Momentum")
    returns_mom.to_excel(writer, sheet_name="Returns_Momentum")
    fav_w_bm.to_excel(writer,    sheet_name="Favorable_BM")
    unfav_w_bm.to_excel(writer,  sheet_name="Unfavorable_BM")
    returns_bm.to_excel(writer,  sheet_name="Returns_BM")
    signal_weights_jan.to_excel(writer, sheet_name="Signal_Weighted")

