import os
import pandas as pd

# Load columns
columns = pd.read_csv("columns.csv")["column id"].to_list()

# Col number
col_sgn = input("Input feature: ")

# Get relevant columns
rel_cols = [col for col in columns if col_sgn in col]
print(rel_cols)

# Get chosen column input
chs = input("Chosen column index (type all or an interger): ")

# Pick chosen column
col_chs = rel_cols[:] if chs == "all" else [rel_cols[int(chs)]]
print("Chosen column: ", col_chs)

# Load chosen data
df_out = pd.read_csv("ukb25909.csv", usecols=["eid", *col_chs])

# Dropna?
if bool(int(input("Drop na? (0 or 1)> "))):
    how = input("How? (any or all)> ")
    df_out = df_out \
        .dropna(how="all", subset=df_out.columns[1:]) \
        .reset_index(drop=True)

# Feature name
feat_name = input("Feature name (make it short and no space!): ")

# Rename columns
df_out.columns = ["eid"] + \
    [feat_name + "_" + str(item.split("-")[1].split(".")[0]) \
     for item in df_out.columns[1:]]

# Save separately as csv
filename = feat_name + ".csv"
df_out.to_csv("feats/" + filename)
