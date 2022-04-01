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

# raise

# %%
# Settings
contrast_A="diab"
contrast_B="age"

# Open data
A = pd.read_csv(SRCDIR + f"pub_meta_volume_stats_{contrast_A}_46.csv", index_col=0) #.iloc[-10:, :].reset_index() #TODO
B = pd.read_csv(SRCDIR + f"pub_meta_volume_stats_{contrast_B}_46.csv", index_col=0) #.iloc[-10:, :].reset_index()

# Nuemrify interval data (from str to list of floats)
numerify_cols = lambda df: df.assign(
        **{
                "conf_int": lambda df: df["conf_int"].apply(
    lambda item: [float(val) for val in item[1:-1].split(" ") if val not in [" ", ""]]),
               "sample_sizes": lambda df: df["sample_sizes"].apply(
    lambda item: [int(val) for val in item[1:-1].split(", ") if val not in [" ", ""]])
    })

A = numerify_cols(A)
B = numerify_cols(B)

# Add contraast columns. DO not edit these. They are used downstream!
A["contrast"] = contrast_A
B["contrast"] = contrast_B

# Normalize age effects to t2dm
B["beta"] = B["beta"]*1

# Merge
df = pd.concat([B, A])

# Make a column with unique index
df["index"] = np.arange(len(df))

# Reorder per value
order = df \
    .query(f'contrast == "{contrast_B}"') \
    .sort_values(by="beta")["index"] \
    .pipe(lambda df:
        pd.concat([df, df+df.shape[0]], axis=0)) \
    .to_frame() \
    .pipe(lambda df:
        df.assign(**{
                "ref": np.arange(df.shape[0])
                }))

df = df.merge(order, on="index", how="inner").sort_values(by="ref")

# For later use
sub_dfs = [df.query(f'contrast == "{contrast_B}"').reset_index(drop=True),
           df.query(f'contrast == "{contrast_A}"').reset_index(drop=True)]

# =============================================================================
# Plot
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Colors
colors_A = colors_from_values(
        np.array(list(-A["beta"]) + [A["beta"].min(), A["beta"].max()]),
        "seismic_r")[:-2]

colors_B = colors_from_values(
        np.array(list(-B["beta"]) + [B["beta"].min() + 0, B["beta"].max()]),
        "PiYG")[:-2]

colors = np.concatenate((colors_B, colors_A), axis=0)

colors_dict = {i: colors[i] for i in range(len(colors))}

# Format labels
df["label"] = df["label"].str.replace("_", " ")

# Plotting
f = sns.FacetGrid(data=df, col="contrast", height=7.25, aspect=1,
                  col_order=[contrast_B, contrast_A], sharex=False, despine=False) \
    .map_dataframe(sns.barplot, y="label", x="beta",
                   palette=colors_dict, hue="index", dodge=False,
                   linewidth=lw*0.5, edgecolor="black",
                   zorder=3, orient="h")

# Formatting
# =======

# For both axes
# ------

# Axis titles
ss = [A["sample_sizes"][0], B["sample_sizes"][0]]
title_texts = [
        f"Age (HC only)\nN$_{{}}$={ss[1][0]:,}", \
        f"T2DM (T2DM+ vs. HC)\nN$_{{T2DM+}}$={ss[0][1]:,}, " \
        f"N$_{{HC}}$={ss[0][0]:,}",
        ]


# Loop through
for i, ax in enumerate(f.axes[0]):

    # Unpack
    sub_df = sub_dfs[i]

    # Remove title
    ax.set_title(title_texts[i])

    # Add grid
    ax.grid(zorder=0, linewidth=0.25*lw)

    # Add x=0 axvline
    ax.axvline(x=0, linewidth=0.75*lw, color="black")

    # Add in errobars
    for _, item in tqdm(enumerate(sub_df.iterrows()), total=len(sub_df)):

        y = item[0]
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

        xoffset = 0.15 if x == 0 else 0.02
        xtext = min(conf_int) + xoffset if x < 0 else max(conf_int) - xoffset
        ytext = y + 0.3

        ax.annotate("   " + text + "   ", xy=[xtext, ytext],
                     ha=ha, va="center", fontsize=8*fs, fontweight="bold")

    # Format bars
    for bar in ax.patches:
        w=0.7
        y = bar.get_y()
        bar.set_y(y + (0.8 - w)/2)
        bar.set_height(w)

    # Add spines
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(0.75*lw)
        ax.spines[sp].set_color("black")


# Axis specific formatting
# ========

# Age
# -----
ax = f.axes[0][0]
ax.spines['right'].set_visible(False)
ax.set_xlim([-1.3, 0.70])
ax.set_xlabel("Percentage change in\ngray matter volume\nacross age (% per year)")

# T2DM
# ----
ax = f.axes[0][1]
ax.set_xlim([-10, 5])
ax.set_xlabel("Percentage difference in\ngray matter volume\nT2DM+ vs. HC (%)")

# Figure formatting
# ------

# Set figure size
plt.gcf().set_size_inches(7.25, 9)

# Add common suptitle
plt.suptitle("Region Specific Gray Matter Volume Changes\n Associated with Age and T2DM: " \
             "UK Biobank Dataset", va="top", y=0.985)

## Add common x label
#plt.gcf().text(0.6, 0.03, "Change In Gray Matter Volume (%)", ha='center',
#        fontsize=14*fs, fontweight="bold")


plt.tight_layout(rect=[0, 0.00, 1, 0.995])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_volume.pdf",
            transparent=True)
plt.close("all")

#from scipy import stats
#stats.pearsonr(B["beta"], A["beta"])
#plt.scatter(B["beta"], A["beta"])
#plt.xlim([-0.75, 0])
#plt.ylim([-3, 0])
