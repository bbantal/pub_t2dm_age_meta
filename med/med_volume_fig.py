#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:40:49 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'med')
get_ipython().run_line_magic('matplotlib', 'inline')


# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/med/volume/"

# Inputs
CTRS = "metfonly_unmed"  # Contrast: diab or age
CTRS_label = "Medicated (T2DM+, metformin only) vs. Unmedicated (T2DM+)"

PARC = 46  # Type of parcellation to use, options: 46 or 139

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Load age results for sorting reference
age_order = pd \
        .read_csv(OUTDIR + "../../volume/stats/pub_meta_volume_stats_age_46.csv",
                       index_col=0) \
        .sort_values(by="beta") \
        .reset_index(drop=True) \
        .reset_index()[["label", "index"]] \
        .rename({"index": "order"}, axis=1)


# Load df
df = pd \
    .read_csv(OUTDIR + f"stats/pub_meta_med_volume_stats_{CTRS}_{PARC}.csv",
              index_col=0) \
    .merge(age_order, on="label") \
    .sort_values(by="order") \
    .reset_index(drop=True) \
    .pipe(lambda df:
        df.assign(**{
                "label": df["label"].str.replace("_", " "),
                "index": np.arange(len(df)),
                "conf_int": lambda df: df["conf_int"].apply(
                        lambda item: [float(val) for val in item[1:-1].split(" ") \
                                      if val not in [" ", ""]]),
                "sample_sizes": lambda df: df["sample_sizes"].apply(
                        lambda item: [int(val) for val in item[1:-1].split(", ") \
                                      if val not in [" ", ""]])
                })) \


# Colors
colors = colors_from_values(
        np.array(list(-df["beta"]) + [df["beta"].min(), df["beta"].max()]),
        "PRGn_r")[:-2]

colors_dict = {i: colors[i] for i in range(len(colors))}

# Make figure
plt.figure(figsize=(7.25, 9))

# Plot
sns.barplot(data=df, y="label", x="beta",
            palette=colors_dict, hue="index", dodge=False,
            linewidth=lw*0.25, edgecolor="black",
            zorder=3, orient="h")

# Turn legend off
plt.legend([],[], frameon=False)

# Get current axis
ax = plt.gca()

# Add grid
ax.grid(zorder=0, linewidth=0.25*lw)

# Add x=0 axvline
ax.axvline(x=0, linewidth=0.75*lw, color="black")

# Add in errobars
for _, item in tqdm(enumerate(df.iterrows()), total=len(df)):

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
    ytext = y + 0.2

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

# Adjust ylim t make space for arrow
plt.gca().set_ylim([item[0] + item[1] for item in zip(plt.gca().get_ylim(), [-3, -2])])


# Add arrow representing directionality
ax.annotate("Improvement\nwith medication",
            xy=(0.8, 0.96), xycoords='axes fraction',
            xytext=(0.2, 0.96), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="fancy, head_width=1.5, head_length=3, tail_width=1.2",
                            connectionstyle="arc3",
                            facecolor='lightgrey',
                            linewidth=1),
            va="center", ha="center", fontsize=9
            )

# Add labels
ss = df["sample_sizes"]
ax.set_xlabel("Percentage difference in gray matter volume\nmedicated vs" \
              " unmedicated (% of avg)", x=0.4)

# Suptitle
plt.suptitle("Region Specific Gray Matter Volume \n" \
          f"{CTRS_label}\n" \
          + f"N$_{{metf+}}$={ss[0][1]:,}, N$_{{metf-}}$={ss[0][0]:,}",
          va="top")

# ttl = plt.title("Region Specific Gray Matter Volume \n" \
#           f"({CTRS_label})\n" \
#           + f"N$_{{metf+}}$={ss[0][1]:,}, N$_{{metf-}}$={ss[0][0]:,}")
# ttl.set_x(ttl.get_position()[0]-0.18)


ax.set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(OUTDIR + f"figures/JAMA_meta_figure_med_volume_{CTRS}.pdf",
            transparent=True)
plt.close("all")
