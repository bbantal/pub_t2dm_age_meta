#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:44:08 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib as mpl
from IPython import get_ipython
from tqdm import tqdm

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, match_cont, \
check_assumptions, detrender
from helpers.data_loader import DataLoader
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================
# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/volume/stats/"
OUTDIR = HOMEDIR + "results/volume/"

EXTRA = "_sex"

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%
crs = ["age", "age", "diab", "diab"]
sts = ["F", "M", "F", "M"]

# Open data
dfs = [pd.read_csv(SRCDIR + f"pub_meta_volume_stats_{item[0]}_46_{item[1]}.csv",
                   index_col=0) for item in zip(crs, sts)]

# Numerify interval data (from str to list of floats)
numerify_cols = lambda df: df.assign(
        **{
                "conf_int": lambda df: df["conf_int"].apply(
    lambda item: [float(val) for val in item[1:-1].split(" ") if val not in [" ", ""]]),
               "sample_sizes": lambda df: df["sample_sizes"].apply(
    lambda item: [int(val) for val in item[1:-1].split(", ") if val not in [" ", ""]])
    })

dfs = map(numerify_cols, dfs)

# Assign set specific columns
dfs = [item[0].assign(**{"contrast": item[1], "sex": item[2]}) for item in zip(dfs, crs, sts)]

# Merge
df = pd.concat(dfs)

# Make a column with unique index
df["index"] = np.arange(len(df))

# Reorder per beta value
df = df \
    .query(f'(contrast=="age") & (sex=="F")') \
    .sort_values(by="beta") \
    ["index"] \
    .pipe(lambda df: pd.concat([df + len(df)*i for i in range(4)], axis=0)) \
    .to_frame() \
    .pipe(lambda df:
        df.assign(**{
                "ref": np.arange(df.shape[0])
                })) \
    .merge(df, on="index", how="inner") \
    .sort_values(by="ref")

# Split up into separate dfs for plotting later
split_dfs = lambda item: df.query(f'(contrast=="{item[0]}") & (sex=="{item[1]}")') \
    .reset_index(drop=True)

sub_dfs = list(map(split_dfs, zip(crs, sts)))


# %%
# =============================================================================
# Plot
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs


# Colors
color_maps = ["Blues", "Reds", "Blues", "Reds"]
colors_set = [colors_from_values(
        np.array(list(-sub_df["beta"]) + [sub_df["beta"].min(), sub_df["beta"].max()]),
        color_map)[:-2] for (sub_df, color_map) in zip(sub_dfs, color_maps)]

colors = np.concatenate(colors_set, axis=0)

colors_dict = {i: colors[i] for i in range(len(colors))}

# Format labels
df["label"] = df["label"].str.replace("_", " ")

# Plotting
f = sns.FacetGrid(data=df, col="contrast", height=7.25, aspect=1,
                  sharex=False, despine=False) \
    .map_dataframe(sns.barplot, y="label", x="beta",
                   hue="sex", hue_order=["F", "M"], dodge=True,
                   linewidth=lw*0.5, edgecolor="black",
                   palette=sns.color_palette(["indianred", "dodgerblue"]),
                   zorder=3, orient="h") \
    .add_legend(bbox_to_anchor=(0.2, 0.88), bbox_transform=plt.gcf().transFigure)

# Formatting
# =======

# For both axes
# ------

# Axis titles
ss = [sub_df["sample_sizes"][0] for sub_df in sub_dfs]
title_texts = [
        f"Age (HC only)\nN$_{{Female}}$={ss[0][0]:,}, " \
        f"N$_{{Male}}$={ss[1][0]:,}",
        f"T2DM (T2DM+ vs HC)\nN$_{{Female}}$={ss[2][0]:,}, " \
        f"N$_{{Male}}$={ss[3][0]:,}"
        ]

# Legend labels
for t, l in zip(f._legend.texts, ["Females", "Males"]):
    t.set_text(l)

# Loop through
for i, (cr, st) in enumerate(zip(crs, sts)):

    # Unpack data
    sub_df = sub_dfs[i]

    # Axis-corresponding index
    aix = int(i/2%2)

    # Current axis
    ax = f.axes[0][aix]

    # Remove title
    ax.set_title(title_texts[aix])

    # Add grid
    ax.grid(zorder=0, linewidth=0.25*lw)

    # Add x=0 axvline
    ax.axvline(x=0, linewidth=0.75*lw, color="black")

    # Add in errobars
    for _, item in tqdm(enumerate(sub_df.iterrows()), total=len(sub_df)):

        y = item[0] - 0.20 + 0.20*(i%2)*2
        ss, p, x, conf_int  = item[1][["sample_sizes", "pval", "beta", "conf_int"]]

        conf_dist = abs(x - np.array(conf_int))[:, None]

        ax.errorbar(x, y, xerr=conf_dist, capsize=1.2*lw, capthick=lw*0.5,
                     elinewidth=lw*0.5, color="black", zorder=100)

    #    text = pformat(p) + p2star(p ) if PRINT_P \
    #            else p2star(p)
    #    ha = "left"
    #    x = 3.2

        text = p2star(p)

        ha = "right" if x < 0 else "left"

        xoffset = 0 #0.5 if x == 0 else 0.5
        xtext = min(conf_int) + xoffset if x < 0 else max(conf_int) - xoffset
        ytext = y + 0.3

        ax.annotate("   " + text + "   ", xy=[xtext, ytext],
                     ha=ha, va="center", fontsize=8*fs, fontweight="bold")


    # # Format bars
    # for bar in ax.patches:
    #     w=0.7
    #     y = bar.get_y()
    #     bar.set_y(y + (0.8 - w)/2)
    #     bar.set_height(w)

    # Add spines d
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(0.75*lw)
        ax.spines[sp].set_color("black")

# Axis specific formatting
# ========

# Age
# -----
ax = f.axes[0][0]
ax.set_xlim([-1.6, 0.70])
ax.spines['right'].set_visible(False)
ax.set_xlabel("Percentage change in\ngray matter volume\nacross age (% per year)")

# T2DM
# ----
ax = f.axes[0][1]
ax.set_xlim([-11, 6])
ax.set_xlabel("Percentage difference in\ngray matter volume\nT2DM+ vs. HC (%)")

# Figure formatting
# ------

# Set figure size
plt.gcf().set_size_inches(7.25, 9)

# Add common suptitle
plt.suptitle("Region Specific Gray Matter Volume Changes Associated with Age and T2DM,\n" \
             "Quantified Separately for Sexes, UK Biobank", va="top", y=0.985)

## Add common x label
#plt.gcf().text(0.6, 0.03, "Change In Gray Matter Volume (%)", ha='center',
#        fontsize=14*fs, fontweight="bold")


plt.tight_layout(rect=[0, 0.00, 1, 0.995])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume_sex_strat.pdf",
            transparent=True)
plt.close("all")

#from scipy import stats
#stats.pearsonr(B["beta"], A["beta"])
#plt.scatter(B["beta"], A["beta"])
#plt.xlim([-0.75, 0])
#plt.ylim([-3, 0])
