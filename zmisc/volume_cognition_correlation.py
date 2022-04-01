#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9 10:47:00 2021

@author: botond

This is a temporary script looking at correlations between structural and cognitive features.

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
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match_mah, match_cont, check_assumptions
get_ipython().run_line_magic('cd', 'volume')

# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
CTRS = "diab"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
PARC = 46  # Type of parcellation to use, options: 46 or 139
## to exlucde due to abnormal total gray matter volumes
RLD = True  # Reload regressor matrices instead of computing them again

# Plan:
# Load volume results - Done
# Parcel them - Done
# Load cognitive results - Done
# Merge them - Done
# Load regressors - TODO
# Merge with data - TODO
# Do a match_mah - TODO
# Run correlations  - Done


# Load volume data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "volume/volume_data.csv").drop(["age", "gender"], axis=1)

# Load labels
labels = pd \
    .read_csv(SRCDIR + "volume/volume_labels.csv",
                     index_col=0, header=0, names=["ID", "label"]) \
    .set_index("ID").to_dict()["label"]

# Load head size normalization factor
head_norm = pd.read_csv(SRCDIR + "volume/head_size_norm_fact.csv")
head_norm = data.merge(head_norm, on="eid", how="inner")[["eid", "norm_fact"]]

# Rename columns
data = data.rename(labels, axis=1).set_index("eid")

# 46 parcellation
# -------
if PARC == 46:

    # Parcellation data
    parc_data = pd.read_csv(SRCDIR + "atlas/46/139_to_46_indexes.csv", index_col=0)

    # Make a code book
    col_names = np.array(data.columns)

    # New dataframe for merged features
    data_extracted = pd.DataFrame()

    # Iterate over all groups of regions
    for item in parc_data.iterrows():

        # Indexes belonging to current group of regions
        indexes = [int(val)+24 for val in item[1]["index"][1:-1].split(", ")]

        # Get corresponding labels
        current_col_names = col_names[indexes]

        # Name of the current group of regions
        name = item[1]["label"]

        # Merge and construct current group of regions
        current_group = data[current_col_names].dropna(how="any").sum(axis=1).rename(name)

        # Normalize with head size
        data_extracted[name] = current_group/(head_norm.set_index("eid")["norm_fact"])

#        print(name, current_col_names)

#    # Add regressors to data
#    df = data_extracted.merge(regressors_matched, on="eid", how="inner")
#
#    # Extract list of features from new feat groups
#    features = list(data_extracted.columns)
#
#    # Exclude regions that are too small
#    features = [feat for feat in features if feat not in excl_region]

data_vol = data_extracted.reset_index().set_index("eid")

# Load cognitive data
# -------
# Labels
labels = {
     "4282-2.0": "Short_Term_Memory",
     "6350-2.0": "Executive_Function",
     "20016-2.0": "Abstract_Reasoning",
     "20023-2.0": "Reaction_Time",
     "23324-2.0": "Processing_Speed",
     }

# Load data
data = pd \
    .read_csv(SRCDIR + "cognition/cognition_data.csv") \
    [["eid",
      *labels.keys()
     ]] \
    .rename(labels, axis=1) \
    .set_index("eid") \
#    .pipe(lambda df: df.assign(**{
#        "Short_Term_Memory": df["Short_Term_Memory"],
#         "Executive_Function": -1*df["Executive_Function"],
#         "Abstract_Reasoning": df["Abstract_Reasoning"],
#         "Reaction_Time": -1*df["Reaction_Time"],
#         "Processing_Speed": df["Processing_Speed"]
#         })) \
#    .pipe(lambda df:
#        ((df - df.mean(axis=0))/df.std(axis=0)).mean(axis=1)) \
#    .rename("score") \
#    .reset_index() \

# Rename columns
data_cogn = data #.rename(labels, axis=1)

df = data_cogn.reset_index().merge(data_vol.reset_index(), on="eid").dropna().set_index(["eid"])

# Take corr
from scipy import stats

#stats.spearmanr(df["Ventral_Striatum"], df["score"])

df_corrs = np.zeros((data_cogn.shape[1], data_vol.shape[1]))

for i, col1 in enumerate(data_cogn.columns):
    for j, col2 in enumerate(data_vol.columns):
        df_corrs[i, j] = stats.spearmanr(df[col1], df[col2])[0]


df_corrs = pd.DataFrame(df_corrs)

df_corrs = df_corrs.set_axis(data_cogn.columns).T.set_axis(data_vol.columns).T

df_corrs.to_csv("/Users/botond/Desktop/volcogn_corrs.csv")

# %%
# Correlation plot
plt.figure(figsize=(6.4, 5.1))
fs=1
#plt.rcParams['xtick.labelsize']=16
#plt.rcParams['ytick.labelsize']=16
#plt.title(f"Correlation based similarities among effects and datasets")
sns.heatmap(df_corrs, vmin=-1, vmax=1, cmap="seismic", annot=False,
            fmt="", linewidth=1, linecolor="k",
            annot_kws={"fontsize": 8*fs})
plt.xticks(rotation=45, ha="right");
plt.tight_layout()
plt.savefig("/Users/botond/Desktop/volcogn_corrplot.pdf")