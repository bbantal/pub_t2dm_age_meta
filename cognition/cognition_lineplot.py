#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:57:36 2021

@author: botond
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
from helpers.regression_helpers import check_covariance, match, check_assumptions
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'cognition')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
CTRS = "diab"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
RLD = 1  # Reload regressor matrices instead of computing them again

print("\nRELOADING REGRESSORS!\n") if RLD else ...

# raise

# %%
# =============================================================================
# Load data
# =============================================================================


# Load cognitive data
# -------
# Labels
labels = {
     "4282-2.0": "Numeric_Memory",
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
     ]]

# Rename columns
data = data.rename(labels, axis=1)

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
# Build regressor matrices
# =============================================================================

# Status
print(f"Transforming.")

# Choose variables
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, sex, college, age_onset]
        ) \
        .drop("age_onset", axis=1)

# Get regressor columns for later
reg_cols = regressors.columns

# Merge domain data and clean out invalied entries
data_merged = data[data>0].dropna()

# Get data columns for later
data_cols = data_merged.columns

# Merge regressors with data
regressors_y = regressors.merge(data_merged, on="eid", how="inner")

# Drop data columns
regressors_clean = regressors_y.drop(data_cols[1:], axis=1)

# Match
if RLD == False:
    # Match
    regressors_matched = match(
            df=regressors_clean,
            main_var="diab",
            vars_to_match=["age", "sex", "college"],
            N=1,
            random_state=100
            )

# Save matched regressors matrix
if RLD == False:
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + "regressors/pub_meta_cognition_lineplot_matched_" \
                f"regressors_{CTRS}.csv")

# =============================================================================
# Statistics
# =============================================================================

# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_cognition_lineplot_matched_regressors_" \
        f"{CTRS}.csv",
        index_col=0)

# Merge and standardize
df = regressors_matched \
    .merge(data_merged, on="eid") \
    .set_index(list(reg_cols)) \
    .pipe(lambda df: df.assign(**{
        "Numeric_Memory": df["Numeric_Memory"],
         "Executive_Function": -1*df["Executive_Function"],
         "Abstract_Reasoning": df["Abstract_Reasoning"],
         "Reaction_Time": -1*df["Reaction_Time"],
         "Processing_Speed": df["Processing_Speed"]
         })) \
    .pipe(lambda df:
        ((df - df.mean(axis=0))/df.std(axis=0)).mean(axis=1)) \
    .rename("score") \
    .reset_index() \

# Linear regression
model = smf.ols(f"score ~ diab + age + C(sex) + C(college)", data=df)
results = model.fit()
print(results.summary())

# Covariance matrix of coefficients
print(results.cov_params())

"""
# CI for the ratio below is computed using an online tool (Fieller method):
https://www.graphpad.com/quickcalcs/errorProp1/?Format=SEM

An alternative approach would be to bootstrap using sigmas and covariances.
"""

# Estimated age gap between the two cohorts#
print(f'\nEstimated age gap: {results.params["diab"]/results.params["age"]:.2f}, years.')

# %%
# =============================================================================
# Plot
# =============================================================================

# Status
print("Plotting.")

# Prep
# -----
# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
lw = lw

# Graphing df with age groups
gdf = df \
    [["score", "age", "diab"]] \
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
sns.lineplot(data=gdf, x="age_group", y="score",
         hue="diab", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=3*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Format
# ----

# Title
plt.title(f"Cognitive Performance across Age\nand T2DM Status\n" \
          f"N$_{{T2DM+}}$={ss[1]:,}, " \
          f"N$_{{HC}}$={ss[0]:,}", x=0.42)


plt.xlabel("Age group (year)")
plt.ylabel("Standardized cognitive performance score")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(handles=legend_handles[::-1],
           labels=["T2DM+", "HC"],
           loc=1)

plt.xticks(rotation=45)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.75*lw)
    plt.gca().spines[sp].set_color("black")

# Annotate stats
tval, pval = results.tvalues["diab"], results.pvalues["diab"]
text = f"T2DM+ vs HC:\nT={tval:.1f}, {pformat(pval)}{p2star(pval)}"
plt.annotate(text, xycoords="axes fraction", xy=[0.279, 0.1],
             fontsize=8*fs, fontweight="regular", ha="center")

plt.gca().yaxis.grid(True)
plt.tight_layout()



# Save
# ----

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_cognition_lineplot.pdf",
            transparent=True)
plt.close("all")
