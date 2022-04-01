#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:47:10 2021

@author: botond

Notes:
-this script looks at how disease duration is distributed across the life span
among subjects with T2DM

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
# With no hue
# =============================================================================

#  Load regressors
# -------

# Src
src = "volume/regressors/" \
             "pub_meta_volume_matched_regressors_diab.csv"

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# %%
#  Check relationship
# ---------

# Transform
df = regressors \
    .query("diab == 1") \
    .merge(age_onset, on="eid") \
    .dropna() \
    .pipe(lambda df:
        df.assign(**{"duration": df["age"] - df["age_onset"]
        }))

# Linear stats
corr = stats.pearsonr(df["age"], df["duration"])

# Plot
sns.lineplot(data=df, x="age", y="duration")

# Format
plt.title("T2DM Duration vs. Age")
text = f"Pearson's r: r={corr[0]:.3f}, p={corr[1]:.2e}, n={df.shape[0]}" + \
        f"\nsource file: {src[:40]}\n{src[40:]}"
plt.annotate(text, xy=[0.05, 0.85], xycoords="axes fraction")
plt.tight_layout()

# Save
# plt.savefig(OUTDIR + "zmix/pub_meta_t2dm-duration-age.pdf")

# Close
# plt.close()

# %%
# =============================================================================
# With hue
# =============================================================================

# Load data
# ------

# Src
src = "volume/regressors/" \
    "pub_meta_volume_matched_regressors_diab.csv"

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# %%
#  Check relationship
# ---------

# Transform
df = regressors \
    .query("diab == 1") \
    .merge(age_onset, on="eid") \
    .dropna() \
    .pipe(lambda df:
        df.assign(**{"duration": df["age"] - df["age_onset"],
                     "sex": df["sex"].apply(lambda item: "M" if item==1 else "F")
        }))


# Linear stats
corrs = [stats.pearsonr(df["age"], df["duration"]) for df in [df.groupby("sex").get_group(i) for i in ["F", "M"]]]
ss = [df.shape[0] for df in [df.groupby("sex").get_group(i) for i in ["F", "M"]]]

# Plot
sns.lineplot(
    data=df, x="age", y="duration",
    hue="sex", hue_order=["F", "M"],
    palette=sns.color_palette(["indianred", "dodgerblue"])
    )

# Format
plt.title("T2DM Duration vs. Age | Sex")
text = f"Pearson's r:" \
        f"\nF: r={corrs[0][0]:.3f}, p={corrs[0][1]:.2e}, n={ss[0]}" \
        f"\nM: r={corrs[1][0]:.3f}, p={corrs[1][1]:.2e}, n={ss[1]}" \
        f"\nsource file: {src[:40]}\n{src[40:]}"
plt.annotate(text, xy=[0.05, 0.8], xycoords="axes fraction")
plt.tight_layout()

# Save
# plt.savefig(OUTDIR + "zmix/pub_meta_t2dm-duration-age-sex.pdf")

# Close
# plt.close()

# %%
# =============================================================================
# Show sample sizes
# =============================================================================

# Load data
# ------

# Src
src = "volume/regressors/" \
    "pub_meta_volume_matched_regressors_diab.csv"

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Transform
# ---------

# Transform
df = regressors \
    .query("diab == 1") \
    .merge(age_onset, on="eid") \
    .dropna() \
    .pipe(lambda df:
        df.assign(**{"duration": df["age"] - df["age_onset"],
                     "sex": df["sex"].apply(lambda item: "M" if item==1 else "F")
        }))

# Show counts
# -----

# Plot
plt.title("Sample size distribution @ Age | Sex")
sns.histplot(
    data=df, x="age", hue="sex", multiple="dodge", hue_order=["F", "M"],
    palette=sns.color_palette(["indianred", "dodgerblue"])
    )
plt.tight_layout()

# Save
# plt.savefig(OUTDIR + "zmix/pub_meta_t2dm_samplesize-age-sex2.pdf")

# Close
# plt.close()


# %%
# Sex specific 2d density plot
# ----

# Title dictionary
title_dict = {"F": "Female", "M": "Male"}

# Plot
g = sns.FacetGrid(data=df, col="sex", col_order=["F", "M"],
              hue="sex", palette=sns.color_palette(["dodgerblue", "indianred"]),) \
    .map_dataframe(
        sns.histplot, "age", "duration", multiple="dodge", hue_order=["F", "M"]
        ) \
    .set_xlabels("Age (year)") \
    .set_ylabels("T2DM Disease Duration (year)") \


# Resize
g.fig.set_size_inches((5, 4))

# Adjust axes (titles, lim)
for ax in g.axes[0]:
    ax.set_title("Females" if "F" == ax.get_title()[-1] else \
                 "Males" if "M" == ax.get_title()[-1] else "", fontsize=12)
    ax.set_xlim([50, 80])
    ax.grid(False)

# Suptitle
g.fig.suptitle("T2DM Disease Distribution across Sample\n" \
               "for Gray Matter Volume Analyses, UK Biobank", fontsize=13)

# Add spines
for ax in g.axes[0]:
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(0.75)
        ax.spines[sp].set_color("black")


plt.tight_layout()

# Save
plt.savefig(OUTDIR + "zmix/pub_meta_t2dm_duration-age-sex_2d.pdf")

# Close
# plt.close()

# tdf = detrender(df=df, x="age", y="duration", subvar="sex", subvar_value=1, weight_fact=2)

# %%
# =============================================================================
# Duration plotter, new version
# =============================================================================

# Load data
# ------

# Src
src = "volume/regressors/" \
    "pub_meta_volume_lineplot_matched_regressors_diab_sex.csv"

# Load regressor matrix for specific case
regressors = pd.read_csv(OUTDIR + src, index_col=0)

# Transform
df = regressors \
    .query("diab == 1") \
    .dropna() \
    .pipe(lambda df:
        df.assign(**{"duration": df["age"] - df["age_onset"],
                     "sex": df["sex"].apply(lambda item: "M" if item==1 else "F")
        }))


# Sample sizes
# ----

# Plot
plt.title("Sample size distribution @ Age | Sex")
sns.histplot(
    data=df, x="age", hue="sex", multiple="dodge", hue_order=["F", "M"],
    palette=sns.color_palette(["indianred", "dodgerblue"])
    )
plt.tight_layout()

# Duration vs age
# -----

# Linear stats
corrs = [stats.pearsonr(df["age"], df["duration"]) for df in [df.groupby("sex").get_group(i) for i in ["F", "M"]]]
ss = [df.shape[0] for df in [df.groupby("sex").get_group(i) for i in ["F", "M"]]]

# Plot
plt.figure()
sns.lineplot(
    data=df, x="age", y="duration",
    hue="sex", hue_order=["F", "M"],
    palette=sns.color_palette(["indianred", "dodgerblue"])
    )

# Format
plt.title("T2DM Duration vs. Age | Sex")
text = f"Pearson's r:" \
        f"\nF: r={corrs[0][0]:.3f}, p={corrs[0][1]:.2e}, n={ss[0]}" \
        f"\nM: r={corrs[1][0]:.3f}, p={corrs[1][1]:.2e}, n={ss[1]}" \
        f"\nsource file: {src[:40]}\n{src[40:]}"
plt.annotate(text, xy=[0.05, 0.8], xycoords="axes fraction")
plt.tight_layout()

# Save
# plt.savefig(OUTDIR + "zmix/pub_meta_t2dm-duration-age-sex.pdf")

# Close
# plt.close()
