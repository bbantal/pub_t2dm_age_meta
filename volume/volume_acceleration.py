#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:13:40 2021

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
from helpers.data_loader import DataLoader
from helpers.regression_helpers import check_covariance, match_cont, check_assumptions
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
CTRS = "duration"  # Contrast: diab or age
T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects
DUR_CO = 10  # Year to separate subjects along duration <x, >=x
PARC = 46  # Type of parcellation to use, options: 46 or 139
RLD = 0 # Reload regressor matrices instead of computing them again

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
# Build regressor matrix
# =============================================================================

# Status
print(f"Building regressor matrix with contrast [{CTRS}]")


# Choose variables and group per duration
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, mp, hrt, htn, age_onset]
        ) \
        .drop(["age_onset", "mp", "hrt"], axis=1) \
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
y = data['Volume of grey matter (normalised for head size)'].rename("feat").reset_index()
regressors_y = y.merge(regressors, on="eid", how="inner")

# Drop feat column
regressors_clean = regressors_y.drop(["feat"], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_volume_acceleration_full_regressors_{CTRS}.csv")

# Interactions among independent variables
var_dict = {
        "age": "cont",
        "sex": "disc",
        "college": "disc",
        }

for name, type_ in var_dict.items():

    if RLD == False:

        # check_covariance(
        #         regressors_clean.query(f'{CTRS} == {CTRS}'),
        #         var1=CTRS,
        #         var2=name,
        #         type1="cont",
        #         type2=type_,
        #         save=True,
        #         prefix=OUTDIR + "covariance/pub_meta_volume_acceleration_covar"
        #         )

        # plt.close("all")


        # Match
        regressors_matched = match_cont(
                df=regressors_clean,
                main_vars=["duration_group"],
                vars_to_match=["age", "sex", "college", "htn"],
                value=1000,
                random_state=10
                )

if RLD == False:
    # Save matched regressors matrix
    regressors_matched \
        .reset_index(drop=True) \
        .to_csv(OUTDIR + f"regressors/pub_meta_volume_acceleration_matched_regressors_{CTRS}.csv")

# <><><><><><><><>
raise
# <><><><><><><><>

# %%
# =============================================================================
# Sample sizes
# =============================================================================

# CTRS specific settings
dc = 1 if CTRS == "diab" else 0
ylim = 120

# Load regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_acceleration_matched_regressors_{CTRS}.csv"
        )

# Figure
plt.figure(figsize=(5, 4))

# Plot
sns.histplot(data=regressors_matched.query(f'diab=={dc}'),
             x="age", hue="sex",
             multiple="stack", bins=np.arange(50, 85, 5),
             palette=["crimson", "royalblue"], alpha=1, zorder=2)

# Annotate total sample size and mean age
text = f"N={regressors_matched.query(f'diab=={dc}').shape[0]:,}"
text = text + " (T2DM+)" if CTRS == "diab" else text
text = text + f"\nMean age={regressors_matched.query(f'diab=={dc}')['age'].mean():.1f}y"
plt.annotate(text, xy=[0.66, 0.88], xycoords="axes fraction", fontsize=10, va="center")

# Legend
legend_handles = plt.gca().get_legend().legendHandles
plt.legend(handles=legend_handles, labels=["Females", "Males"], loc=2,
           fontsize=8)

# Formatting
plt.xlabel("Age")
plt.ylim([0, ylim])
plt.grid(zorder=1)
# plt.title("Gray Matter Volume", fontsize=10)

# Save
plt.tight_layout(rect=[0, 0.00, 1, 0.85])
plt.savefig(OUTDIR + f"stats_misc/pub_meta_volume_acceleration_sample_sizes_{CTRS}.pdf",
            transparent=True)

# Close all
# plt.close("all")

# %%
# =============================================================================
# Statistics
# =============================================================================

# Prep
# ------
# Get regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_acceleration_matched_regressors_{CTRS}.csv",
        index_col=0)

# Join regressors with data
y = data['Volume of grey matter (normalised for head size)'].rename("Whole_Brain").reset_index()
df = regressors_matched.merge(y, on="eid", how="inner")

# Take nondiabetic subs
sdf = df.query('diab == 1')

# Fit
# ------
# Feature
feat = "Whole_Brain"

# Fit the model to get brain age
model = smf.ols(f"{feat} ~ age + C(sex) + C(college) + htn + duration", data=sdf)
results = model.fit()

# Monitor
# --------

# print(results.summary())

# Save detailed stats report
with open(OUTDIR + f"stats_misc/pub_meta_volume_acceleration_regression_table_{feat}" \
          f"_{CTRS}.html", "w") as f:
    f.write(results.summary().as_html())

# Check assumptions
check_assumptions(
        results,
        sdf,
        prefix=OUTDIR + \
        f"stats_misc/pub_meta_volume_acceleration_stats_assumptions_{feat}_{CTRS}"
        )

# Results
# -------

# Calculate acceleration of aging in additional year/year
acc_res = results.params["duration"]/results.params["age"]

# Print regression estimates
print(results.summary())

# Covariance matrix of coefficients
print(results.cov_params())

"""
# CI for the ratio is computed using an online tool (Fieller method):
https://www.graphpad.com/quickcalcs/errorProp1/?Format=SEM

An alternative approach would be to bootstrap using sigmas and covariances.
"""

print(f"Acceleration: +{acc_res:.2g} year/year", f'p={results.pvalues["duration"]:.2g},', \
      "significant" if results.pvalues["duration"]<0.05 else "not significant!")



# %%
# =============================================================================
# Visualize
# =============================================================================

# Prep
# -----
# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs
lw = lw*1.0

# Get data
gdf = df.copy()

# Make age groups
gdf = gdf \
    .pipe(lambda df:
        df.assign(**{"age_group": pd.cut(df["age"], np.arange(0, 100, 5),
               include_lowest=True, precision=0).astype(str)}))# \
    # .query('age_group not in ["(40, 45]", "(45, 50]", "(75, 80]"]')

# Sort
gdf = gdf.sort_values(by=["age", "duration"], na_position="first")

# Sample sizes
print("Sampe sizes, age info:\n", gdf.groupby(['duration_group', 'age_group'])["age"].describe())

# Colors
palette = sns.color_palette(["black", "darkorange", "red"])

# Content
# -----

# Make figure
plt.figure(figsize=(5, 4))

# Create plot
sns.lineplot(data=gdf, x="age_group", y=feat,
         hue="duration_group", ci=68, err_style="bars",
         marker="o", linewidth=1*lw, markersize=2*lw, err_kws={"capsize": 2*lw,
                                                         "capthick": 1*lw,
                                                         "elinewidth": 1*lw},
         sort=False, palette=palette)

# Annotate stats
tval, pval = results.tvalues["duration"], results.pvalues["duration"]
text = (f"T2DM disease duration\nas a continuous linear factor:\n"
       # f"${{H_0}}$:  $\mathrm{{\\beta}}$${{_t}}$ = 0\n" \
       # f"${{H_1}}$:  $\mathrm{{\\beta}}$${{_t}}$ ≠ 0\n" \
       f"T = {tval:.1f}; {pformat(pval)}{p2star(pval)}")

plt.annotate(text, xycoords="axes fraction", xy=[0.3, 0.1],
             fontsize=8*fs, fontweight="regular", ha="center")


# Format
# ----

# Title
ttl = plt.title("Gray Matter Atrophy across Age and T2DM Disease Duration:\n" \
          f"UK Biobank Dataset, "
          f"N$_{{≥10y}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{0–9y}}$={int(gdf.shape[0]/3)}, " \
          f"N$_{{HC}}$={int(gdf.shape[0]/3)}"
          )
ttl.set_x(ttl.get_position()[0]-0.056)

plt.xlabel("Age group (year)")
#plt.ylabel("Gray matter volume delineated\nbrain age (y)")

plt.ylabel("Gray matter volume\n(mm3, normalized for headsize)")
plt.gca().yaxis.set_major_formatter(mtc.FuncFormatter
       (lambda x, pos: f"{x/1e5:.1f}"))
plt.annotate("×10$^5$", xy=[-0.09, .98], xycoords="axes fraction",
             fontsize=8*fs, va="center")

legend_handles, _ = plt.gca().get_legend_handles_labels()
[ha.set_linewidth(5) for ha in legend_handles]

plt.legend(title="T2DM disease duration",
           handles=legend_handles,# [::-1],
           labels=["HC", "0-9 years", "≥10 years"],
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

plt.tight_layout(rect=[0.05, 0., 0.95, 0.99])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_acceleration.pdf",
            transparent=True)
plt.close("all")
