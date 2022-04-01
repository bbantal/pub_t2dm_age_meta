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
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, \
check_assumptions, detrend_diab_sex, detrend_diab_sex_info
from helpers.data_loader import DataLoader
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
EXTRA = "_sex"
T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects
RLD = 0

print("\nRELOADING REGRESSORS!\n") if RLD else ...

# <><><><><><><><>
# raise
# <><><><><><><><>

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

# Initiate loader object
dl = DataLoader()

# Load data
dl.load_basic_vars(SRCDIR)

# Extract relevant variables
age, sex, diab, college, bmi, mp, hrt, age_onset, duration, htn = \
    (dl.age, dl.sex, dl.diab, dl.college, dl.bmi, dl.mp, dl.hrt, dl.age_onset, \
    dl.duration, dl.htn)


# Restrictive variables
# -----

# Perform filtering
dl.filter_vars(AGE_CO, T1DM_CO)

# Extract filtered series
age, mp, hrt, age_onset = dl.age, dl.mp, dl.hrt, dl.age_onset


# %%
# =============================================================================
# Transform
# =============================================================================

# Status
print("Transforming.")


# Merge IVs and put previously defined exclusions into action (through inner merge)
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, mp, hrt, htn, age_onset, duration]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)

# Merge regressors with data to keep records only with existing data
regressors_y = regressors.merge(data, on="eid", how="inner")

# Drop data columns
regressors_clean = regressors_y.drop("volume", axis=1)

# Match
if RLD == False:

    # Balance out duration as well
    # ------
    regressors_detrended = detrend_diab_sex(regressors_clean, thr=.05)

    # Match
    regressors_matched = match(
        df=regressors_detrended,
        main_vars=["sex", "diab"],
        vars_to_match=["age", "college", "htn"],
        random_state=11
        )

    # Look at balance
    detrend_diab_sex_info(regressors_matched)

# Save matched regressors matrix
if RLD == False:
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + "regressors/pub_meta_volume_lineplot_matched_" \
                f"regressors_{CTRS}{EXTRA}.csv")

# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_lineplot_matched_regressors_" \
        f"{CTRS}{EXTRA}.csv",
        index_col=0)

# %%
# =============================================================================
# Statistics
# =============================================================================

# Status
print("Statistics.")

# Descriptive stats
# ---------

# Make copy
df = regressors_matched.copy()

# Duration
print("Median duration:\n", df.groupby(["sex", "diab"])["duration"].median())
print("Mean duration:\n", df.groupby(["sex", "diab"])["duration"].mean())

# Sample sizes
ss = df \
    .pipe(lambda df: df.assign(**{
        "age_group":pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
            })) \
    .groupby(["age_group"])["eid"].count() \
    .divide(4)

print("Sample sizes from age groups:\n", ss)

# ss.plot.bar()
# plt.xticks(rotation=45)
# plt.ylabel("Count")

# Inferential stats
# ----------

# Merge regressors with data
df = regressors_matched \
    .merge(data, on="eid")

# Separately for sexes
# >>>>>>>>

# Labels for printing
labels = ["Female", "Male"]

# It
for sex_val in [0, 1]:

    # Take relevant subset (F or M only)
    sdf = df.query(f'sex=={sex_val}')

    # Linear regression
    model = smf.ols("volume ~ diab + age + C(college) + htn", data=sdf)
    results = model.fit()
    # print(results.summary())

    print(f'\n>>>>>>\nCohort: {labels[sex_val]}')
    print(
        f'Coeffs: T2DM={results.params["diab"]:.0f}, ' \
        f'Age={results.params["age"]:.0f}'
            )
    print(
        f'Estimated age gap: {results.params["diab"]/results.params["age"]:.2f} years.'
    )

    # Covariance matrix of coefficients
    # print(results.cov_params())

# Interaction
# >>>>>>>>>

# Make a copy
sdf = df.copy()

# Linear regression
model = smf.ols("volume ~ diab + sex + diab*sex + age + C(college) + htn", data=sdf)
results = model.fit()
# print(results.summary())

print(
  f'\nInteraction term (separate analysis): coeff:' \
  f'{results.params["diab:sex"]:.0f}, p value: {results.pvalues["diab:sex"]:.2g}'
      )

# Covariance matrix of coefficients
# print(results.cov_params())



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
print("Plotting.")

# Prep
# -----

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
lw = lw*1

# Graphing df:
# add back in data
# unify columns of contrast variables
# make age groups
gdf = regressors_matched \
    .merge(data, on="eid") \
    .pipe(lambda df: df.assign(**{
        "group": df[["sex", "diab"]].astype(str).agg("_".join, axis=1),
        "age_group":pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
            })) \
    .sort_values(by=["age", "group"]) \
    .query('age_group not in ["(40, 45]"]') \
    .query('age_group not in ["(40, 45]", "(45, 50]"]') \
#    .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')


# Sample sizes
ss = regressors_matched.groupby(["sex", "diab"])["eid"].count().to_list()[0]

# Colors
palette = sns.color_palette(["coral", "maroon", "dodgerblue", "navy"])

# Content
# -----

# Make figure
plt.figure(figsize=(5, 4))

# Create plot
sns.lineplot(data=gdf, x="age_group", y="volume",
         hue="group", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=3*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Format
# ----

# Title
plt.title("Gray Matter Volume vs Age, T2DM Status and Sex\n" \
          f"N={ss} (Per Group, Exact Matched)", x=0.42, y=1.05)

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter volume delineated\nbrain age (y)")

plt.ylabel("Gray matter volume\n(mm3, normalized for headsize)")
plt.gca().yaxis.set_major_formatter(mtc.FuncFormatter
       (lambda x, pos: f"{x/1e5:.1f}"))
plt.annotate("×10$^5$", xy=[0, 1.03], xycoords="axes fraction",
             fontsize=8*fs, va="center")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(handles=legend_handles,
           labels=["F, HC", "F, T2DM+", "M, HC", "M, T2DM+"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()
# plt.xticks(rotation=45)

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.75*lw)
    plt.gca().spines[sp].set_color("black")

# Annotate stats
# tval, pval = results.tvalues["diab"], results.pvalues["diab"]
# text = f"T2DM+ vs HC:\nT={tval:.1f}, {pformat(pval)}{p2star(pval)}"
# plt.annotate(text, xycoords="axes fraction", xy=[0.40, 0.08],
#              fontsize=8*fs, fontweight="regular", ha="center")

plt.gca().yaxis.grid(True)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUTDIR + f"figures/JAMA_meta_figure_volume_lineplot{EXTRA}.pdf",
            transparent=True)
# plt.close("all")

