#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:18:29 2021

@author: benett
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================
plt.style.use("ggplot")
sns.set_style("whitegrid")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

# raise

# %%
# =============================================================================
# Load data
# =============================================================================

#  Load design matrix for volume within t2dm matched only
# -------

# Modality
mod = "volume"

# Src
src = f"{mod}/regressors/pub_meta_{mod}_full_regressors_diab.csv"

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Load ivs
# ------

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Open mp
mp = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
   [["eid", "mp_2"]] \
    .rename({"mp_2": "mp"}, axis=1)
    # .query('(mp in [0,  1]) | (mp!=mp)')

# %%
# =============================================================================
# Show distributions
# =============================================================================

# Merge for total sample
df = regressors.merge(mp, how="left")


# Plot
g = sns.FacetGrid(data=df, col="diab", col_order=[0, 1],
              hue="mp", hue_order=[1, 0],
              palette=sns.color_palette(["dodgerblue", "indianred"]),
              sharey=False) \
    .map_dataframe(
        sns.histplot, "age", multiple="dodge",
        binwidth=1
        ) \
    .set_titles("diab: {col_name}") \
    .add_legend()

plt.tight_layout()

# Save
plt.savefig(OUTDIR + "zmix/pub_meta_age-mp-t2dm.pdf")

# Close
plt.close()

# %%
# =============================================================================
# Show sample size loss for different scenarios
# =============================================================================

# Baseline case: no exclusion
df_0 = regressors \
    .merge(mp, how="left") \
    .query('diab == 1') \
    .reset_index()

# Case A: exclude subjects based on mp 0/1
mask = df_0.merge(mp, how="left").eval('(mp in [1] & sex == 0) | (mp!=mp & sex == 1)')
df_A = df_0[mask]
df_A

df_0["A"] = mask

# Case B: exclude based on age cutoff
mask = df_0.eval('age > 50')
df_B = df_0[mask]
df_B

df_0["B"] = mask

# Stack for representation
df_0_A = df_0 \
    .pipe(lambda df:
          df.assign(**{
              "retain": df.apply(lambda row: \
           "inc, M" if ((row["mp"] != row["mp"]) & (row["sex"] == 1)) \
           else "inc, mp+" if ((row["mp"] == 1) & (row["sex"] == 0)) \
           else "exc, mp-" if (row["mp"]  == 0) & (row["sex"] == 0) \
           else "exc, mp else" if (row["mp"]  == row["mp"]) & (row["sex"] == 0) \
           else "exc, mp missing" if (row["mp"] != row["mp"]) & (row["sex"] == 0) \
           else "other", axis=1),
            "subset": "exclusion based on mp"}))

df_0_B = df_0 \
    .pipe(lambda df:
          df.assign(**{
              "retain": df.apply(lambda row: \
           "inc, M" if (row["sex"] == 1) \
           else "inc, F" if ((row["age"] > 55) & (row["sex"] == 0)) \
           else "exc, age", axis=1),
              "subset": "exclusion based on age"}))

df_0_stacked = pd.concat([df_0_A, df_0_B], axis=0)

# Sample sizes
ss_A = df_0_A.value_counts(subset="retain")
ss_B = df_0_B.value_counts(subset="retain")

# Plot
g = sns.FacetGrid(data=df_0_stacked, col="subset",
              hue="retain", sharey=True,
              palette=sns.color_palette(["dodgerblue", "coral", "darkred", "yellow", "gray", "coral", "green"])) \
    .map_dataframe(
        sns.histplot, "age",
        binwidth=1,
        binrange=[45, 80],
        ) \
    .set_titles("subset: {col_name}") \
    .add_legend(loc=1)

# plt.tight_layout()

# Save
# plt.savefig(OUTDIR + "zmix/pub_meta_mp_exclusion_comparison.pdf")

# Close
# plt.close()

df_0_C = df_0_A.query('~(((age < 60) & (mp != 1)) & sex == 0)')

# Save sample sizes
plt.figure(figsize=(4, 2))
plt.axis('off')

# Annotate sample sizes
plt.annotate(df_0_A["subset"][0] + "\n" + str(ss_A),
             xy=[0.1, 0.8], va="top", xycoords="figure fraction")
plt.annotate(df_0_A["subset"][1] + "\n" + str(ss_B),
             xy=[0.6, 0.8], va="top", xycoords="figure fraction")

# Save
plt.savefig(OUTDIR + "zmix/pub_meta_mp_exclusion_comparison_numbers.pdf")

# =============================================================================
# Combining mp reports across instances
# =============================================================================

mp_0 = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
   .merge(regressors, on="eid", how="right") \
   .query('diab == 1 & sex == 0')

mp_0.query('(mp_2 != 1) & ((mp_1 == 1) | (mp_0 == 1))').shape

mp = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
   .merge(regressors, on="eid", how="right") \
   .query('diab == 1 & sex == 0') \
   .pipe(lambda df: df.assign(**{
       "mp": df.eval('mp_0 == 1 | mp_1 == 1 | mp_2 == 1').astype(int), \
       "mp_2": df.eval('mp_2 == 1').astype(int)})) \
   [["eid", "age", "mp", "mp_2"]]

mp["mp+"] = mp["mp"] + mp["mp_2"]

sns.histplot(data=mp, x="age", hue="mp+", multiple="stack",
             binwidth=1, binrange=[45, 80],
             palette=sns.color_palette(["dodgerblue", "indianred", "green"]))

# %%
# =============================================================================
# Age at last menopause
# =============================================================================

# Open age of last mp
mp_age = pd \
    .read_csv(SRCDIR + "ivs/mp_age.csv", index_col=0) \
   # [["eid", "mp_age_2"]] \
   #  .rename({"mp_2": "mp"}, axis=1)

# Merge with regressors
df_0 = regressors \
    .merge(mp_age, how="left") \
    .query('diab == 1 & sex == 0')


# %%
# Step0: total variance

plt.figure(figsize=(8, 6))
plt.hist(mp_age["mp_age_2"], bins=np.arange(0, 70, 2))
plt.xlabel("age at mp")
plt.ylabel("count")

plt.figure(figsize=(8, 6))
sns.histplot(data=mp_age, x="mp_age_2", cumulative=True, stat='density', element="step", fill=False)
plt.xlabel("age at mp")
plt.ylabel("cumulative probability")

# %%
# Step1: get a sense of availability within F, within T2DM only

df_0.shape
df.shape

# Query for availability
df = df_0.query('mp_age_0 == mp_age_0 | mp_age_1 == mp_age_1 | mp_age_2 == mp_age_2')
df.shape

df = df_0.query('mp_age_0 == mp_age_0')
df.shape

df = df_0.query('mp_age_1 == mp_age_1')
df.shape

df = df_0.query('mp_age_2 == mp_age_2')
df.shape

# Query for availability
df = df_0.query('mp_age_0 > 0 | mp_age_1 > 0 | mp_age_2 > 0')
df.shape

df = df_0.query('mp_age_0 > 0')
df.shape

df = df_0.query('mp_age_1 > 0')
df.shape

df = df_0.query('mp_age_2 > 0')
df.shape

df = df_0.query('mp_age_0 > 0 | mp_age_2 > 0')
df.shape

"""
instance 2 have most samples, but combining with other instances would yield more samples.
"""

# %%
# Step 2: Check contradictions
df_0.query('mp_age_2 > age')


"""
No contradictions in instance 2.
"""

# %%
# Step3: variance
tempdf = mp_age.copy()
tempdf["diff"] = mp_age["mp_age_2"] - mp_age["mp_age_0"]
plt.hist(tempdf["diff"])

# %%
# Step4/5: relation to mp, let's see if we can get more samples out of this
mp_age = pd \
    .read_csv(SRCDIR + "ivs/mp_age.csv", index_col=0)

mp = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0)

df_0 = regressors \
    .merge(mp_age, how="left") \
    .merge(mp, how="left") \
    .query('diab == 1 & sex == 0')

df = df_0.query('~(mp_age_0 > 0) & (mp_0 == 1)')
df.shape

df = df_0.query('(mp_age_0 > 0) & ~(mp_0 == 1)')
df.shape

df = df_0.query('~(mp_age_1 > 0) & (mp_1 == 1)')
df.shape

df = df_0.query('(mp_age_1 > 0) & ~(mp_1 == 1)')
df.shape

df = df_0.query('~(mp_age_2 > 0) & (mp_2 == 1)')
df.shape

df = df_0.query('(mp_age_2 > 0) & ~(mp_2 == 1)')
df.shape

df = df_0.query('~((mp_age_0 > 0) | (mp_age_1 > 0) | (mp_age_2 > 0)) & ((mp_0 == 1) | (mp_1 == 1) | (mp_2 == 1))')
df.shape

df = df_0.query('((mp_age_0 > 0) | (mp_age_1 > 0) | (mp_age_2 > 0)) & ~((mp_0 == 1) | (mp_1 == 1) | (mp_2 == 1))')
df.shape

"""
There is no records where age of menopause would bring in the record as postmenopausal.
"""

"""
What would save about 18 samples is if I combined mp_0, mp_1, mp_2 -> TODO
"""

# %%
# Step6: duration of mp

df = df_0.query('mp_age_2 > 0')

df["mp_duration"] = df["age"] - df["mp_age_2"]

sns.histplot(data=df, x="age", y="mp_duration", cbar=True, binwidth=[1, 1])

# %%
# =============================================================================
# HRT
# =============================================================================
hrt = pd \
    .read_csv(SRCDIR + "ivs/hrt_age.csv", index_col=0)

df = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
   .pipe(lambda df: df.assign(**{
       "mp": df.eval('mp_0 == 1 | mp_1 == 1 | mp_2 == 1').astype(int), \
       "mp_2": df.eval('mp_2 == 1').astype(int)})) \
   [["eid", "mp", "mp_2"]] \
   .merge(regressors, on="eid", how="right") \
   .query('diab == 1 & sex == 0') \
   .merge(hrt, on="eid", how="left")

df.query('mp == 1 & hrt_age_2 == -11')
tdf = df.query('(mp == 1) & (hrt_age_2 > 0)')

plt.hist(tdf["hrt_age_2"], bins=np.arange(0, 100, 5))
plt.xlabel("age at last HRT (pm+ only)")
plt.ylabel("count")

plt.hist(tdf["age"] - tdf["hrt_age_2"], bins=np.arange(0, 100, 5))
plt.xlabel("time since last HRT (pm+ only)")
plt.ylabel("count")
