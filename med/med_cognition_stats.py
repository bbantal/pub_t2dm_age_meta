#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:41:58 2021

@author: botond

Notes:
The structure of this script is determined by the modality (for the most part)
and by contrast to some extent.

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
from helpers.data_loader import DataLoader
get_ipython().run_line_magic('cd', 'med')

# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/med/cognition/"

# Inputs
CTRS = "metfonly_unmed"  # Contrast: diab or age
T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects
excl_sub = []  # SUbjects to exlude, if any
RLD = False  # Reload regressor matrices instead of computing them again

# <><><><><><><><>
# raise
# <><><><><><><><>

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
     "23324-2.0": "Processing_Speed"
     }

# Load data
data = pd \
    .read_csv(SRCDIR + "cognition/cognition_data.csv") \
    [["eid",
      *labels.keys()
     ]]

# Rename columns
data = data.rename(labels, axis=1)

# Exclude subjects
#data = data.query(f'eid not in {excl_sub}')

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
# Load medication specific data
med = pd.read_csv(SRCDIR + f"med/{CTRS}.csv")[["eid", CTRS]]

# %%
# =============================================================================
# Build regressor matrices
# =============================================================================

# Status
print(f"Building regressor matrices with contrast [{CTRS}].")

# Merge IVs and put previously defined exclusions into action (through merge)
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, bmi, mp, hrt, htn, med, age_onset, duration]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)

# Assign labels to features
features = [col for col in data.columns if "eid" not in col]

# Iterate over all features
for i, feat in enumerate(features):

    # Status
    print(f"Current feature: {feat}")

    # Clean feature
    # -----
    # Extract data, drop non-positive records
    data_feat = data.query(f'`{feat}` > 0')[["eid", feat]]

    # Constrain ivs to only those samples that have y values
    regressors_y = regressors \
        .merge(data_feat, on="eid", how="inner") \
        .dropna(how="any") \

    # Temporarily remove y values
    regressors_clean = regressors_y.drop(feat, axis=1)

    # Coarse ivs before matching
    age_bins = np.arange(0, 100, 5)
    duration_bins = np.arange(0, 100, 3)

    regressors_clean = regressors_clean.pipe(lambda df: df.assign(**{
        "age_group": pd.cut(df["age"], age_bins, include_lowest=True,
                            labels=age_bins[1:]),
        "duration_group": pd.cut(df["duration"], duration_bins, include_lowest=True,
                                 labels=duration_bins[1:]),
            }))

    # Save full regressor matrix
    regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_med_cognition_full_" \
                            f"regressors_{CTRS}.csv")

    # Check regressoriance of ivs
    # -----

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "sex": "disc",
            "college": "disc",
            "bmi": "cont",
            "duration": "cont"
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1=CTRS,
                var2=name,
                type1="disc",
                type2=type_,
                save=True,
                prefix=OUTDIR + f"covariance/pub_meta_med_cognition_covar_{feat}_"
                )

        plt.close("all")

    # Perform matching
    # ------
    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_vars=[CTRS],
                vars_to_match=["age_group", "sex", "college", "htn", "duration_group"],
                random_state=1
                )

    # Save matched regressors matrix
    if RLD == False:
        regressors_matched.to_csv(OUTDIR + \
            f"regressors/pub_meta_med_cognition_matched_regressors_{feat}_{CTRS}.csv")

# Reset style
plt.close("all")
plt.style.use("default")
plt.rcdefaults()

# <><><><><><><><>
# raise
# <><><><><><><><>


# %%
# =============================================================================
# Sample sizes
# =============================================================================

# Set style
from helpers.plotting_style import plot_pars, plot_funcs

# Iterate over all features
for i, feat in enumerate(features):

    # Load regressors
    regressors_matched = pd.read_csv(OUTDIR + \
            f"regressors/pub_meta_med_cognition_matched_regressors_{feat}_{CTRS}.csv",
            index_col=0)

    # Figure
    plt.figure(figsize=(3.5, 2.25))

    # Plot
    sns.histplot(data=regressors_matched.query(f'{CTRS}==1'),
                 x="age", hue="sex",
                 multiple="stack", bins=np.arange(50, 85, 5),
                 palette=["crimson", "royalblue"], alpha=1, zorder=2)

    # Annotate total sample size
    text = f"N={regressors_matched.query(f'{CTRS}==1').shape[0]}"
    text = text + " (T2DM+)" if CTRS == "diab" else text
    plt.annotate(text, xy=[0.66, 0.9], xycoords="axes fraction", fontsize=7)

    # Legend
    legend_handles = plt.gca().get_legend().legendHandles
    plt.legend(handles=legend_handles, labels=["Females", "Males"], loc=2,
               fontsize=8)

    # Formatting
    plt.xlabel("Age")
    plt.ylim([0, 80])
    plt.grid(zorder=1)
    plt.title(f'{feat.replace("_", " ")}', fontsize=10)

    # Save
    plt.tight_layout(rect=[0, 0.00, 1, 0.995])
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_med_cognition_sample_sizes_{feat}" \
                f"_{CTRS}.pdf",
                transparent=True)

# Reset style
plt.close("all")
plt.style.use("default")
plt.rcdefaults()

# %%
# =============================================================================
# Fit models
# =============================================================================

# Status
print(f"Fitting models for contrast [{CTRS}].")

# Dict to store stats that will be computed below
feat_stats = {}

# Iterate over all features
for i, feat in enumerate(features):

    # Status
    print(f"Current feature: {feat}")

    # Clean feature
    # -----
    # Extract data, drop non-positive records
    data_feat = data.query(f'`{feat}` > 0')[["eid", feat]]

    # Load regressors
    regressors_matched = pd.read_csv(OUTDIR + \
            f"regressors/pub_meta_med_cognition_matched_regressors_{feat}_{CTRS}.csv",
            index_col=0)

    # Reunite regressors and y values of the current feature
    sdf = regressors_matched.merge(data_feat, on="eid")

    # Get sample sizes
    sample_sizes = sdf.groupby(CTRS)["eid"].count()

    # Regression
    # -------
    # Formula
    formula = f"{feat} ~ age + C(sex) + C(college) + C(htn) + bmi + duration + {CTRS}"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # ------
    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_med_cognition_regression_table_{feat}" \
              f"_{CTRS}.html", "w") as f:
        f.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_med_cognition_stats_assumptions_{feat}_{CTRS}"
            )

    # Plot across age
    plt.figure()
    plt.title(feat)
    sns.lineplot(data=sdf[[feat, "age", CTRS]], x="age", y=feat, hue=CTRS,
                 palette=sns.color_palette(["black", "red"]))
    plt.tight_layout()
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_med_cognition_age-{CTRS}-plot_{feat}.pdf")
    plt.close()

    # Plot distribution of feature
    plt.figure()
    plt.title(feat)
    plt.hist(sdf[feat])
    plt.xlabel("score")
    plt.ylabel("count")
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_cognition_dist_{feat}_{CTRS}.pdf")
    plt.tight_layout()
    plt.close()

    # Save results
    # -------
    # Normalization factor
    norm_fact = sdf[feat].mean()/100

    # Get relevant key for regressor
    rel_key = [key for key in results.conf_int().index.to_list() \
           if CTRS in key][0]

    # Get effect size
    tval = results.tvalues.loc[rel_key]
    beta = results.params.loc[rel_key]/norm_fact

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]/norm_fact

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval,
                              pval, beta, np.array(conf_int)]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int"]) \
        .reset_index() \
        .rename({"index": "label"}, axis=1) \
        .assign(**{"pval": lambda df: pg.multicomp(list(df["pval"]),
                                                   method="bonf")[1]})

# Flip tasks where higher score is worse
flip_tasks = ["Executive_Function", "Reaction_Time"]

# Temporarily assign index to df index column so I can slice into it using task names
df_out = df_out.set_index("label")

# Slice into df and multiply respective tasks with -1
df_out.loc[flip_tasks, ["tval", "beta", "conf_int"]] = \
    df_out.loc[flip_tasks, ["tval", "beta", "conf_int"]]*-1

# Undo to original indexing
df_out = df_out.reset_index()

# Save outputs
df_out.to_csv(OUTDIR + f"stats/pub_meta_med_cognition_stats_{CTRS}.csv")
