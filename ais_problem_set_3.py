import pandas as pd

def fix_index(df):
    df = df.copy()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.set_index('Date',inplace=True)
    return df



###  Data Loading

df = pd.read_excel("data/Problem_Set_3_Data.xlsx")
momentum_df = fix_index(df.iloc[:, 0:7])
BM_df = fix_index(df.iloc[:, [0] + list(range(8, 14))])
returns_df = fix_index(df.iloc[:, [0] + list(range(15, 21))])

