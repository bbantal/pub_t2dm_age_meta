#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:20:43 2021

@author: botond

Notes:
This script involves a statistical analysis focusing on the acceleration of
cognitive decline among subjects with T2DM.

"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, check_assumptions
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'cognition')

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
CTRS = "age"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
#RLD = True  # Reload regressor matrices instead of computing them again


# %%
# =============================================================================
# Load data
# =============================================================================

# Load age info
# ------
# Age
age = pd.read_csv(SRCDIR + "ivs/age.csv", index_col=0)[["eid", "age-0", "age-2"]] \
    .dropna(how="any")

# Diabetes diagnosis
diab = pd.read_csv(SRCDIR + "ivs/diab.csv", index_col=0)[["eid", "diab-0", "diab-2"]] \
    .dropna(how="any")

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Age of onset + some filtering as per age of onset, diabetic status
"""
Remove subjects who reported diabetes status (H -> D or D -> H)
Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
"""
age_onset \
    .merge(diab, on="eid", how="inner") \
    .query('`diab-0` == `diab-2`') \
    .query(f'(`diab-0`==0 & age_onset!=age_onset) or (`diab-0`==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# Merge age info
age_info = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, age_onset]
        ) \
        .drop("age_onset", axis=1)

# Transform age info
"""
# Transform to long format
Separate dataframe into two based on whether visit specific or not, using regex
If visit specific, stack out visit to the side
If not visit specific, duplicate, add visit label and then stack out visit label
Then add the separated dfs back together along subject and visit index labels
Rename multiindexes: add visit as label
"""

age_info = age_info \
    .pipe(lambda df:
        pd.concat(
            [df \
                .filter(regex=r'.-[0-9]', axis=1) \
                .pipe(lambda df: df.set_axis(df.columns.str \
                                             .split("-", expand=True),
                                             axis=1)) \
                .stack(),
            df \
                .filter(regex=r'^[^0-9]+$', axis=1) \
                .pipe(lambda df: pd \
                          .concat([df, df], axis=1) \
                          .set_axis([col+ind for ind in ["-0", "-2"] \
                                     for col in df.columns],
                                     axis=1)) \
                .pipe(lambda df: df.set_axis(df.columns.str \
                                             .split("-", expand=True),
                                             axis=1)) \
                .stack()
            ],
            axis=1
                    )
            ) \
    .pipe(lambda df: df.set_index(df.index.set_names(["index", "visit"]))) \
    .reset_index() \
    .drop("index", axis=1)  \
    .set_index(["eid", "visit"])


# Load cognition data
# -------
"""
# Transform cognitive data to long format
Drop misc columns like age and such
Set subject as index
Select predefined cognitive tasks for which instance 0 existed
Change dash to underscore in column names
Stack out visit order for each task
Rename multiindexes: add visit as label
"""

labels = {
     "f4282": "Short_Term_Memory",
     "f20016": "Abstract_Reasoning",
     "f20023": "Reaction_Time",
     }

data = pd \
        .read_csv(SRCDIR + "cognition/cognition_data.csv") \
        .drop(["2443-2.0", "2976-2.0", "6138-2.0", "21022-0.0"], axis=1) \
        .set_index("eid") \
        [["4282-0.0", "4282-2.0",
          "20016-0.0", "20016-2.0",
          "20023-0.0", "20023-2.0"]] \
        .pipe(lambda df: df.set_axis(
                ["f" + col.split("-")[0] + "_" + col.split("-")[1][0] \
                 for col in df.columns],
                 axis=1)) \
        .pipe(lambda df: df.set_axis(df.columns.str.split('_', expand=True),
                                     axis=1)) \
        .stack() \
        .pipe(lambda df: df.set_index(df.index.set_names(["eid", "visit"]))) \
        .rename(labels, axis=1)

# Merge meta and cognitive data
df = pd.concat([age_info, data], axis=1).reset_index()

# Columns containing basic info
info_cols = ["eid", "visit", "age", "diab"]

# Set eid as string
df["eid"] = df.eid.astype(str)

# Features to run stat analysis for
feats = labels.values()

# Itearte over all features
for feat in feats:

    # Extract current feature
    sdf = df[info_cols + [feat]]

    # Drop nans for current feature
    sdf = sdf.dropna(axis=0, how="any", subset=["age", "diab", feat])

    # Extract subjects who have data for both visits
    subs_double = list(sdf.groupby(["eid"]).count().query('visit == 2').index)
    sdf = sdf.query(f'(eid in {subs_double})')

    # Compute the difference between visit 2 and visit 0
    sdf[["diab_diff", "age_diff", "score_diff"]] = \
            sdf.groupby(["eid"])[["diab", "age", feat]].apply(lambda x: x.diff())

    # Get rid of nan rows
    sdf = sdf.dropna(axis=0, how="any", subset=["score_diff"])

    # Get rid of subjects who have changed diabetes status
    sdf = sdf.query('diab_diff == 0')

    # Divide with age
    sdf["slope"] = sdf["score_diff"]/sdf["age_diff"]

    # Get sample sizes
    ss = sdf.groupby(["diab"]).count()["eid"].to_list()

    # Compare means and compute percentage difference
    means = sdf.groupby(["diab"])["slope"].mean()
    perc = abs(means[0] - means[1])/means[0] * 100

    # Separate according to diabetes
    sdfs = [v for k, v in sdf.groupby('diab')]

    # Run t-test
    tval, pval = stats.ttest_ind(sdfs[0]["slope"], sdfs[1]["slope"], equal_var=False)

    # Show results
    p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
    text = \
           f"Feature: {feat}\n" \
           f"T={tval:.1f}\n{pformat(pval)}" \
           f"{p2star(pval)}\n" \
           f"$\mathbf{{N_{{T2DM}}}}$={ss[0]}\n" \
           f"$\mathbf{{N_{{ctrl}}}}$={ss[1]}\n" \
           f"Percentage: {perc:.2f}"

    print(text)