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

get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================
# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/neurofunction/stats/"
OUTDIR = HOMEDIR + "results/neurofunction/"

#raise

# %%
# Settings
#contrast="t2dm"

# Open data
t2 = pd.read_csv(SRCDIR + "pub_meta_neurofunction_roi_stats_diab_46.csv", index_col=0) #.iloc[-10:, :].reset_index()
ag = pd.read_csv(SRCDIR + "pub_meta_neurofunction_roi_stats_age_46.csv", index_col=0) #.iloc[-10:, :].reset_index()

# Nuemrify interval data (from str to list of floats)
numerify_cols = lambda df: df.assign(
        **{
                "conf_int": lambda df: df["conf_int"].apply(
    lambda item: [float(val) for val in item[1:-1].split(" ") if val not in [" ", ""]]),
               "sample_sizes": lambda df: df["sample_sizes"].apply(
    lambda item: [int(val) for val in item[1:-1].split(", ") if val not in [" ", ""]])
    })

t2 = numerify_cols(t2)
ag = numerify_cols(ag)

# Add contraast columns. DO not edit these. They are used downstream!
t2["contrast"] = "t2dm"
ag["contrast"] = "age"

# Normalize age effects to t2dm
ag["beta"] = ag["beta"]*1

# Merge
df = pd.concat([ag, t2])

# Make a column with unique index
df["index"] = np.arange(len(df))

# Reorder per value of t2dm
order = df \
    .query('contrast == "age"') \
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
sub_dfs = [df.query('contrast == "age"').reset_index(drop=True),
           df.query('contrast == "t2dm"').reset_index(drop=True)]

# Or just pick jsut 1 contrast at a time
#df = df.query(f'contrast == "{contrast}"')

# =============================================================================
# Plot
# =============================================================================

# Style
fs=1.3
lw=2
plt.style.use("default")
#plt.style.use("ggplot")

#sns.set_style("whitegrid")

plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['text.color'] = "black"
plt.rcParams['axes.labelcolor'] = "black"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12*fs
plt.rcParams['xtick.labelsize'] = 13*fs
plt.rcParams['ytick.labelsize']=12*fs
plt.rcParams['axes.labelsize']=11*fs
plt.rcParams['axes.labelweight'] = "bold"
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 20*fs
plt.rcParams['text.latex.preamble'] = [r'\boldmath']
plt.rcParams['figure.titlesize'] = 16*fs
plt.rcParams['figure.titleweight'] = "bold"
plt.rcParams['axes.titlesize'] = 13*fs
plt.rcParams['axes.titleweight'] = "bold"


# Astrix
def p2star(p):
    if p > 0.05:
        return ""
    elif p > 0.01:
        return "*"
    elif p > 0.001:
        return "**"
    else:
        return "***"

# Colors
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

colors_t2 = colors_from_values(
        np.array(list(-t2["beta"]) + [t2["beta"].min(), t2["beta"].max()]),
        "seismic_r")[:-2]

colors_age = colors_from_values(
        np.array(list(-ag["beta"]) + [ag["beta"].min() + 0, ag["beta"].max()]),
        "PuOr")[:-2]

colors = np.concatenate((colors_age, colors_t2), axis=0)

colors_dict = {i: colors[i] for i in range(len(colors))}

# Format labels
df["label"] = df["label"].str.replace("_", " ")

# Plotting
f = sns.FacetGrid(data=df, col="contrast", height=16, aspect=0.6,
                  sharex=False, despine=False) \
    .map_dataframe(sns.barplot, y="label", x="beta",
                   palette=colors_dict, hue="index", dodge=False,
                   linewidth=lw, edgecolor="black",
                   zorder=3, orient="h")

# Formatting
# =======

# For both axes
# ------

# Axis titles
ss = [t2["sample_sizes"][0], ag["sample_sizes"][0]]
title_texts = [
        f"Age (T2DM- only, sex-matched)\nN$_{{}}$={ss[1][0]:,}",
        f"T2DM (T2DM+ vs T2DM-, age and sex-matched)\nN$_{{T2DM+}}$={ss[0][1]:,}, " \
        f"N$_{{T2DM-}}$={ss[0][0]:,}"
        ]


# Loop through
for i, ax in enumerate(f.axes[0]):

    # Unpack
    sub_df = sub_dfs[i]

    # Remove title
    ax.set_title(title_texts[i])

    # Add grid
    ax.grid(zorder=0)

    # Add x=0 axvline
    ax.axvline(x=0, linewidth=3, color="black")

    # Add in errobars
    for _, item in tqdm(enumerate(sub_df.iterrows()), total=len(sub_df)):

        y = item[0]
        ss, p, x, conf_int  = item[1][["sample_sizes", "pval", "beta", "conf_int"]]

        conf_dist = abs(x - np.array(conf_int))[:, None]

        ax.errorbar(x, y, xerr=conf_dist, capsize=3, capthick=lw,
                     elinewidth=lw, color="black", zorder=100)

    #    text = pformat(p) + p2star(p ) if PRINT_P \
    #            else p2star(p)
    #    ha = "left"
    #    x = 3.2

        text = p2star(p)

        ha = "right" if x < 0 else "left"

        xoffset = 0 #0.15 if x == 0 else 0.02
        xtext = min(conf_int) + xoffset if x < 0 else max(conf_int) - xoffset
        ytext = y + 0.2

        ax.annotate("   " + text + "   ", xy=[xtext, ytext],
                     ha=ha, va="center", fontsize=12*fs, fontweight="bold")

    # Add spines
    for sp in ['bottom', 'top', 'right', 'left']:
        ax.spines[sp].set_linewidth(2)
        ax.spines[sp].set_color("black")


# Axis specific formatting
# ========

# Age
# -----
ax = f.axes[0][0]
ax.set_xlim([-0.01, 0.01])
ax.spines['right'].set_visible(False)
ax.set_xlabel("Percentage change in ALFF per year (%)")

# T2DM
# ----
ax = f.axes[0][1]
ax.set_xlim([-0.08, 0.08])
ax.set_xlabel("Difference in ALFF in T2DM+ vs T2DM- subjects (%)")

# Figure formatting
# ------

# Add common suptitle
plt.suptitle("Changes in ALFF sssociated with age and T2DM: " \
             "UK Biobank dataset")

## Add common x label
#plt.gcf().text(0.6, 0.03, "Change In Gray Matter Volume (%)", ha='center',
#        fontsize=14*fs, fontweight="bold")


plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_neurofunction_roi.pdf",
            transparent=True)
plt.close("all")

#from scipy import stats
#stats.pearsonr(ag["beta"], t2["beta"])
#plt.scatter(ag["beta"], t2["beta"])
#plt.xlim([-0.75, 0])
#plt.ylim([-3, 0])