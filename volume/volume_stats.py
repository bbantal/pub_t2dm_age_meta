#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:47:00 2020

@author: botond

Notes:
-this script performs linear regression on gray matter volume data
-I normalize with head size, but only when merging regions, not at the
beginning!



"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, match_cont, \
check_assumptions, detrender
from helpers.data_loader import DataLoader
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')

# =============================================================================
# Setup
# =============================================================================

# plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
CTRS = "age"  # Contrast: diab or age
T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects
PARC = 46  # Type of parcellation to use, options: 46 or 139
## to exlucde due to abnormal total gray matter volumes
STRAT_SEX = False # Stratify sex or not #TODO: need to adjust detrending accordinlgy
SEX = 1  # If stratifying per sex, which sex to keep

EXTRA = ""  # Extra suffix for saved files
excl_region = ["Pallidum"]  # Regions to exclude
RLD = False  # Reload regressor matrices instead of computing them again

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

# Merge IVs and put previously defined exclusions into action (through merge)
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, mp, hrt, htn, age_onset, duration]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)

# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# If contrast is sex and we want to separate across age
if CTRS == "sex":
    # Exclude subjects with T2DM OR subjects without T2DM (toggle switch)
    regressors = regressors.query("diab == 0")

# Optional: stratify per sex
if STRAT_SEX:
    # Include subjects of one sex only
    regressors = regressors.query(f"sex == {SEX}")

# Inner merge regressors with a gm column to make sure all included subjects have data
y = data['Volume of grey matter (normalised for head size)'].rename("feat").reset_index()
regressors_y = y.merge(regressors, on="eid", how="inner")

## Fit model
#sdf = regressors_y

#model = smf.ols(f"feat ~ C(diab) + age + C(sex) + C(college)", data=sdf)
#results = model.fit()
#
## Print results
#results.summary()

## Check assumptions
#check_assumptions(results, sdf)

# Drop feat column
regressors_clean = regressors_y.drop(["feat"], axis=1)

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_volume_full_regressors_{CTRS}{EXTRA}.csv") # TODO

if CTRS == "age":

    # # Interactions among independent variables
    # var_dict = {
    #         "sex": "disc",
    #         "college": "disc",
    #         }

    # for name, type_ in var_dict.items():

    #     check_covariance(
    #             regressors_clean,
    #             var1=name,
    #             var2="age",
    #             type1=type_,
    #             type2="cont",
    #             save=True,
    #             prefix=OUTDIR + "covariance/pub_meta_volume_covar"
    #             )

    #     plt.close("all")

    if RLD == False:

        # Match
        regressors_matched = match_cont(
                df=regressors_clean,
                main_vars=["age"],
                vars_to_match=["sex", "college", "htn"],
                value=3,
                random_state=1
                )

if CTRS == "diab":

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "sex": "disc",
            "college": "disc",
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1="diab",
                var2=name,
                type1="disc",
                type2=type_,
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_volume_covar"
                )

        plt.close("all")

    if RLD == False:

        # Match
        regressors_matched = match(
            df=regressors_clean,
            main_vars=["diab"],
            vars_to_match=["age", "sex", "college", "htn"],
            random_state=1
            )


if CTRS == "sex":

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "college": "disc",
            "htn": "disc"
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1="sex",
                var2=name,
                type1="disc",
                type2=type_,
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_volume_covar"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_vars=["sex"],
                vars_to_match=["age", "college", "htn"],
                random_state=1
                )



if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_volume_matched_regressors_{CTRS}{EXTRA}.csv")

# <><><><><><><><>
raise
# <><><><><><><><>

# %%
# =============================================================================
# Sample sizes
# =============================================================================

# If not separating per sex
if ~STRAT_SEX:

    # CTRS specific settings
    dc = 1 if CTRS == "diab" else 0
    ylim = 300 if CTRS == "diab" else 1500

    # Load regressors
    regressors_matched = pd.read_csv(
            OUTDIR + f"regressors/pub_meta_volume_matched_regressors_{CTRS}{EXTRA}.csv"
            )

    # Figure
    plt.figure(figsize=(3.5, 2.25))

    # Plot
    sns.histplot(data=regressors_matched.query(f'diab=={dc}'),
                 x="age", hue="sex",
                 multiple="stack", bins=np.arange(50, 85, 5),
                 palette=["crimson", "royalblue"], alpha=1, zorder=2)

    # Annotate total sample size and mean age
    text = f"N={regressors_matched.query(f'diab=={dc}').shape[0]:,}"
    text = text + " (T2DM+)" if CTRS == "diab" else text
    text = text + f"\nMean age={regressors_matched.query(f'diab=={dc}')['age'].mean():.1f}y"
    plt.annotate(text, xy=[0.66, 0.88], xycoords="axes fraction", fontsize=7, va="center")

    # Legend
    legend_handles = plt.gca().get_legend().legendHandles
    plt.legend(handles=legend_handles, labels=["Females", "Males"], loc=2,
               fontsize=8)

    # Formatting
    plt.xlabel("Age")
    plt.ylim([0, ylim])
    plt.grid(zorder=1)
    plt.title("Gray Matter Volume", fontsize=10)

    # Save
    plt.tight_layout(rect=[0, 0.00, 1, 0.995])
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_volume_sample_sizes_{CTRS}.pdf",
                transparent=True)

    # Close all
    plt.close("all")


# %%
# =============================================================================
# Extract gray matter volumes for regions from raw data
# =============================================================================

# Status
print(f"Extracting volume values with contrast [{CTRS}] at parcellation [{PARC}]")

# Load regressors
regressors_matched = pd.read_csv(
        OUTDIR + f"regressors/pub_meta_volume_matched_regressors_{CTRS}{EXTRA}.csv")

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

    # Add regressors to data
    df = data_extracted.merge(regressors_matched, on="eid", how="inner")

    # Extract list of features from new feat groups
    features = list(data_extracted.columns)

    # Exclude regions that are too small
    features = [feat for feat in features if feat not in excl_region]

# 139 parcellation
# -----
if PARC == 139:

    # Parcellation data
    parc_data = pd \
        .read_csv(SRCDIR + "atlas/139/ukb_gm_labelmask_139.csv") \
        .pipe(lambda df:
            df.assign(**{
              "label": df["label"].apply(lambda item: item \
                             .replace(" ", "_") \
                             .replace(",", "") \
                             .replace("'", "") \
                             .replace("-", ""))
              })) \

    # Assign labels to features
    features = parc_data["label"].to_list()

    # Transform raw gray matter volume data
    df = data \
        .set_axis(list(data.columns[:25]) + features, axis=1) \
        .iloc[:, 25:] \
        .divide(head_norm.set_index("eid")["norm_fact"], axis="index") \
        .merge(regressors_matched, on="eid", how="inner")


# %%
# =============================================================================
# Fit models
# =============================================================================

# Status
print(f"Fitting models for contrast [{CTRS}] at parcellation [{PARC}]\n")

# Set style for plotting
from helpers.plotting_style import plot_pars, plot_funcs
lw=1.5

# Dictionary to store stats
feat_stats = {}

# Iterate over all regions
for i, feat in tqdm(enumerate(features), total=len(features), desc="Models fitted: "):

    # Prep
    # ----
    # Extract current feature
    sdf = df[["eid", "age", "diab", "sex", "college", "htn", f"{feat}"]]

    # Get sample sizes
    sample_sizes = sdf.groupby("sex" if CTRS == "sex" else "diab")["eid"].count()

    # Fit
    # -----
    # Formula
    if CTRS == "age":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(htn)"

    if CTRS == "diab":
        formula = f"{feat} ~ C(diab) + age + C(sex) + C(college) + C(htn)"

    if CTRS == "sex":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(htn)"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # --------

    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_volume_regression_table_{feat}" \
              f"_{CTRS}_{PARC}{EXTRA}.html", "w") as f:
        f.write(results.summary().as_html())
    # print(results.summary())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_volume_stats_assumptions_{feat}_{CTRS}_{PARC}{EXTRA}"
            )

    # Lineplot across age
    if CTRS == "diab":
        gdf = sdf \
            [[feat, "age", "diab"]] \
            .pipe(lambda df: df.assign(**{"age_group":
                    pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
                    })) \
            .sort_values(by="age")

        plt.figure(figsize=(3.5, 3))
        plt.title(feat)
        sns.lineplot(data=gdf, x="age_group", y=feat, hue="diab",
                     palette=sns.color_palette(["black", "red"]),
                     ci=68, err_style="bars", marker="o",
                     linewidth=1*lw, markersize=3 *lw, err_kws={"capsize": 2*lw,
                         "capthick": 1*lw, "elinewidth": 1*lw})
        legend_handles = plt.gca().get_legend().legendHandles
        plt.legend(handles=legend_handles, labels=["HC", "T2DM+"], loc="best",
                   fontsize=8, title="")
        plt.xlabel("Age group")
        plt.ylabel("Gray matter volume\n(mm3, normalized)")
        plt.xticks(rotation=45)
        plt.grid()
        plt.title(feat.replace("_", " "))

        plt.tight_layout()
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_volume_age-diab-plot_{feat}_{PARC}{EXTRA}.pdf",
                    transparent=True)
        plt.close()


    # Save results
    # -------
    # Normalization factor
    if CTRS in ["age", "diab"]:
        norm_fact = sdf.query('diab==0')[feat].mean()/100

    elif CTRS == "sex":
        norm_fact = sdf.query('sex==0')[feat].mean()/100
    else:
        print("Unknown contrast for norm_fact")

    # Get relevant key for regressor
    rel_key = [key for key in results.conf_int().index.to_list() \
           if CTRS in key][0]

    # Get effect size
    tval = results.tvalues.loc[rel_key]
    beta = results.params.loc[rel_key] \
        /norm_fact

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]/norm_fact
    plus_minus = beta - conf_int[0]

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval, pval, beta,
                             np.array(conf_int), plus_minus]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int", "plus_minus"]) \
        .reset_index() \
        .rename({"index": "label"}, axis=1) \
        .assign(**{"pval": lambda df: pg.multicomp(list(df["pval"]),
                                                   method="bonf")[1]})

# Save outputs
df_out.to_csv(OUTDIR + f"stats/pub_meta_volume_stats_{CTRS}_{PARC}{EXTRA}.csv")

