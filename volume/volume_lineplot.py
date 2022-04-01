#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:58:20 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, check_assumptions
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
CTRS = "diab"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
RLD = 1

print("\nRELOADING REGRESSORS!\n") if RLD else ...

# raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load volume data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "volume/volume_data.csv").drop(["age", "gender"], axis=1) \
    [["eid", '25005-2.0']].rename({'25005-2.0': "volume"}, axis=1)

# Load regressors
# ------
# Age
age = pd.read_csv(SRCDIR + "ivs/age.csv", index_col=0)[["eid", "age-2"]] \
    .rename({"age-2": "age"}, axis=1)

# Sex
sex = pd \
    .read_csv(SRCDIR + "ivs/sex.csv", index_col=0)[["eid", "sex"]] \
    .set_axis(["eid", "sex"], axis=1)

# Diabetes diagnosis
diab = pd.read_csv(SRCDIR + "ivs/diab.csv", index_col=0)[["eid", "diab-2"]] \
    .rename({"diab-2": "diab"}, axis=1) \
    .query('diab >= 0')

# College
college = pd.read_csv(SRCDIR + "ivs/college.csv", index_col=0)[["eid", "college"]] \

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Transform
# =============================================================================

# Status
print("Transforming.")


# Choose variables and group per duration
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, age_onset]
        ) \
        .drop("age_onset", axis=1) \

# Merge regressors with data
regressors_y = regressors.merge(data, on="eid", how="inner")

# Drop data columns
regressors_clean = regressors_y.drop("volume", axis=1)

# Match
if RLD == False:
    # Match
    regressors_matched = match(
            df=regressors_clean,
            main_var="diab",
            vars_to_match=["age", "sex", "college"],
            N=1,
            random_state=1
            )

# Save matched regressors matrix
if RLD == False:
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + "regressors/pub_meta_volume_lineplot_matched_" \
                f"regressors_{CTRS}.csv")


# =============================================================================
# Statistics
# =============================================================================

# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_lineplot_matched_regressors_" \
        f"{CTRS}.csv",
        index_col=0)

# Merge regressors with data
df = regressors_matched \
    .merge(data, on="eid") \

# Linear regression
model = smf.ols(f"volume ~ diab + age + C(sex) + C(college)", data=df)
results = model.fit()
print(results.summary())

# Covariance matrix of coefficients
print(results.cov_params())

"""
# CI for the ratio below is computed using an online tool (Fieller method):
https://www.graphpad.com/quickcalcs/errorProp1/?Format=SEM

An alternative approach would be to bootstrap using sigmas and covariances.
"""

# Estimated age gap between the two cohorts
print(f'\nEstimated age gap: {results.params["diab"]/results.params["age"]:.2f}, years.')

# %%
# =============================================================================
# Plot
# =============================================================================

# Status
print(f"Plotting.")

# Prep
# -----

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
lw = lw*1

# Graphing df with age groups
gdf = regressors_matched \
    .merge(data, on="eid") \
    .pipe(lambda df: df.assign(**{"age_group":
            pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
            })) \
    .sort_values(by="age") \
    .query('age_group not in ["(40, 45]"]') \
    .query('age_group not in ["(40, 45]", "(45, 50]"]') \
#    .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')


# Sample sizes
ss = regressors_matched.groupby(["diab"])["eid"].count().to_list()

# Colors
palette = sns.color_palette(["black", "red"])

# Content
# -----

# Make figure
plt.figure(figsize=(3.625, 5))

# Create plot
sns.lineplot(data=gdf, x="age_group", y="volume",
         hue="diab", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=3*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Format
# ----

# Title
plt.title("Gray Matter Volume Across Age\nand T2DM Status\n" \
          f"N$_{{T2DM+}}$={ss[1]:,}, " \
          f"N$_{{HC}}$={ss[0]:,}\n", x=0.42)

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter volume delineated\nbrain age (y)")

plt.ylabel("Gray matter volume (voxel count)")
plt.gca().yaxis.set_major_formatter(mtc.FuncFormatter
       (lambda x, pos: f"{x/1e5:.1f}"))
plt.annotate("×10$^5$", xy=[0, 1.03], xycoords="axes fraction",
             fontsize=8*fs, va="center")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(handles=legend_handles[::-1],
           labels=["T2DM+", "HC"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()
plt.xticks(rotation=45)

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.75*lw)
    plt.gca().spines[sp].set_color("black")

# Annotate stats
tval, pval = results.tvalues["diab"], results.pvalues["diab"]
text = f"T2DM+ vs HC:\nT={tval:.1f}, {pformat(pval)}{p2star(pval)}"
plt.annotate(text, xycoords="axes fraction", xy=[0.40, 0.08],
             fontsize=8*fs, fontweight="regular", ha="center")

plt.gca().yaxis.grid(True)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_lineplot.pdf",
            transparent=True)
plt.close("all")

