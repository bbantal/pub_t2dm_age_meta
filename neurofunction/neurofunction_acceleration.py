#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:13:40 2021

@author: botond

This is a temporary script.

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
from helpers.regression_helpers import check_covariance, match_multi, check_assumptions
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'neurofunction')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
CTRS = "duration"  # Contrast: diab or age
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
DUR_CO = 10  # Year to separate subjects along duration <x, >=x
PARC = 46  # Type of parcellation to use, options: 46 or 139
excl_region = ["Pallidum"]  # Regions to exclude
RLD = False # Reload regressor matrices instead of computing them again

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Load neurofunction data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "neurofunction/avg_alff.csv", index_col=0)

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

# Ses
ses = pd.read_csv(SRCDIR + "ivs/ses.csv", index_col=0)[["eid", "ses"]]

# BMI
bmi = pd.read_csv(SRCDIR + "ivs/bmi.csv", index_col=0)[["eid", "bmi-2"]] \
    .rename({"bmi-2": "bmi"}, axis=1) \
    .dropna(how="any")

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Duration
duration = age_onset \
    .merge(age, on="eid", how="inner") \
    .pipe(lambda df: df.assign(**{
            "duration": df["age"] - df["age_onset"]})) \
    .dropna(how="any") \
    [["eid", "duration"]]

# Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Build regressor matrix
# =============================================================================

# Status
print(f"Building regressor matrix with contrast [{CTRS}]")


# Choose variables and group per duration
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, ses, bmi, diab, age_onset]
        ) \
        .drop("age_onset", axis=1) \
        .merge(duration, on="eid", how="left") \
        .pipe(lambda df: df.assign(**{
        "duration_group":
            df["duration"] \
                .apply(lambda x: \
                       "ctrl" if x!=x else \
                       f"<{DUR_CO}" if x<DUR_CO else \
                       f">={DUR_CO}" if x>=DUR_CO else np.nan)
                }
            ))

# Inner merge regressors with a gm column to make sure all included subjects have data
y = data[["eid", 'alff_mean']].rename({"alff_mean": "feat"}, axis=1).reset_index()
regressors_y = y.merge(regressors, on="eid", how="inner")

# Drop feat column
regressors_clean = regressors_y.drop(["feat"], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_acceleration_" \
                        f"full_regressors_{CTRS}.csv")

# Interactions among independent variables
var_dict = {
        "age": "cont",
        "sex": "disc",
        "college": "disc",
        "ses": "disc",
        "bmi": "cont",
        }

for name, type_ in var_dict.items():

    check_covariance(
            regressors_clean.query(f'{CTRS} == {CTRS}'),
            var1=CTRS,
            var2=name,
            type1="cont",
            type2=type_,
            save=True,
            prefix=OUTDIR + "covariance/pub_meta_neurofunction_acceleration_covar"
            )

    plt.close("all")

if RLD == False:
    # Match
    regressors_matched = match_multi(
            df=regressors_clean,
            main_var="duration_group",
            vars_to_match=["age", "sex", "college"],
            N=1,
            random_state=1
            )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_acceleration_" \
                f"matched_regressors_{CTRS}.csv")

# %%
# =============================================================================
# Statistics
# =============================================================================

# Prep
# ------
# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_neurofunction_acceleration_matched_regressors_{CTRS}.csv",
        index_col=0)

# Join regressors with data
y = data.rename({"alff_mean": "Whole_Brain"}, axis=1).reset_index()
df = regressors_matched.merge(y, on="eid", how="inner")

# Take nondiabetic subs
sdf = df.query('diab == 1')

# Fit
# ------
# Feature
feat = "Whole_Brain"

# Fit the model to get brain age
model = smf.ols(f"{feat} ~ age + C(sex) + C(college) + C(ses) + bmi + duration", data=sdf)
results = model.fit()

# Monitor
# --------

# Save detailed stats report
with open(OUTDIR + f"stats_misc/pub_meta_neurofunction_acceleration_regression_table_{feat}" \
          f"_{CTRS}.html", "w") as f:
    f.write(results.summary().as_html())

# Check assumptions
check_assumptions(
        results,
        sdf,
        prefix=OUTDIR + \
        f"stats_misc/pub_meta_neurofunction_acceleration_stats_assumptions_{feat}_{CTRS}"
        )

# %%
# =============================================================================
# Visualize
# =============================================================================

# Prep
# -----
# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Get data
gdf = df.copy()

# Make age groups
gdf = gdf \
    .pipe(lambda df:
        df.assign(**{"age_group": pd.cut(df["age"], np.arange(0, 100, 5),
               include_lowest=True, precision=0).astype(str)})) \
    .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')

# Sort
gdf = gdf.sort_values(by=["age", "duration"], na_position="first")

# Sample sizes
print("Sampe sizes, age info:\n", gdf.groupby(['duration_group', 'age_group'])["age"].describe())

# Colors
palette = sns.color_palette(["black", "tomato", "darkred"])

# Content
# -----

# Make figure
plt.figure(figsize=(4.25, 5.5))

# Create plot
sns.lineplot(data=gdf, x="age_group", y=feat,
         hue="duration_group", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=2*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Annotate stats
tval, pval = results.tvalues["duration"], results.pvalues["duration"]
text = f"Time since T2DM diagnosis\nas a continuous linear factor:\n" \
       f"$\mathbf{{H_0}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ = 0\n" \
       f"$\mathbf{{H_1}}$:  $\mathrm{{\\beta}}$$\mathbf{{_t}}$ ≠ 0\n" \
       f"T = {tval:.1f}\n{pformat(pval)}" \
       f"{p2star(pval)}"

plt.annotate(text, xycoords="axes fraction", xy=[0.279, 0.03],
             fontsize=8*fs, fontweight="bold", ha="center")


# Format
# ----

# Title
plt.title("Mean ALFF across age and\nT2DM disease progression:\n" \
          f"UK Biobank dataset\n" \
          f"(N$_{{≥10 years}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{0-9 years}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{HC}}$={int(gdf.shape[0]/3)}, "
          "\nage, education and sex-matched)")

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter neurofunction delineated\nbrain age (y)")

plt.ylabel("Mean ALFF")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(title="    Time Since\nT2DM Diagnosis",
           handles=legend_handles[::-1],
           labels=["≥10 years", "0-9 years", "HC"],
           loc=1)

plt.gca().xaxis.tick_bottom()
plt.gca().yaxis.tick_left()

for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.5*lw)
    plt.gca().spines[sp].set_color("black")

plt.gca().xaxis.grid(False)
plt.tight_layout()

# Save
# ----

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_neurofunction_acceleration.pdf",
            transparent=True)
plt.close("all")
