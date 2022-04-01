#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:04:45 2020

@author: botond

Notes:
This script generates a multi-panel figure from cognitive data.

"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
import matplotlib as mpl
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers import plotting_style
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'cognition')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/cognition/"

# Inputs
# Case specific values
cases = ["age_F", "age_M", "diab_F", "diab_M"]
titles = [
        "Age: UK Biobank Dataset (HC only)",
        "T2DM: UK Biobank Dataset (T2DM+ vs. HC)",
        ]

ylabeltexts = [
        "Percentage change in cognitive\nperformance across age (% per year)",
        "Percentage difference in cognitive\nperformance T2DM+ vs. HC (%)",
        ]

x_dodge = 0.18  # Extent of dodge for hue
colors = ["RdBu_r", "RdBu", "RdBu_r", "RdBu"]
ylims = [[-2.5, 0.3], [-2.5, 0.3], [-23.0, 4.5], [-23.0, 4.5]]
sfs = [4e0, 4e0, 1e0, 1e0]  # Marker size factors
sfscf = [5e3, 5e3, 1.5e3, 1.5e3]  # Marker size scale factors
sdxo = [0.76, 0.76, 0.79, 0.79]  # x axis offset of scale info
# sfscf2 = [1e4, 1e4, 1e3, 1e3]  # Marker size scale factors
# sdxo2 = [0.76, 0.76, 0.75, 0.75]  # x axis offset of scale info
cms = [10, 10, 5, 5]  # Colormap scaling to distort - gradient
cmo = [-1, 0, -10, 0]  # Colormap offset - intensity
textpads = [0.05, 0.05, 0.5, 0.5]  # Padding for text along y axis
xtickrots = [0, 0, 0, 0]  # Rotation of xticks
xtickvas = ["top", "top", "top", "top"]  # Vertical alignment for xticks
xtickpads = [0, 0, 0, 0]  # Paddong fo xticks

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%

# Load data
# ------

# Dict to store data
data = {}

# Iterate over all cases (age, diab, meta...)
for case in cases:

    # Load
    df = pd.read_csv(
            OUTDIR + f"stats/pub_meta_cognition_stats_{case}.csv",
            index_col=0
            )

    # Transfomrations specific to cas
    # Cast strings to float arrays
    df["sample_sizes"] = \
        df["sample_sizes"].apply(lambda item: np.array(item[1:-1].split(", "), dtype=int))
    df["conf_int"] = \
        df["conf_int"].apply(lambda item:
                np.array([float(val) for val in item[1:-1].split(" ") if len(val) > 0])
                )

    # Order
    if case == cases[0]:
        order = df.sort_values(by="beta").reset_index(drop=True)["label"]
        order_dict = dict((v,k) for k,v in order.to_dict().items())

    df = df.sort_values(by="label", key=lambda x: x.map(order_dict), ignore_index=True)

    # Assign transformed df to data dict
    data[case] = df


# %%
# =============================================================================
# Figure
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Figure
f = plt.figure(figsize=(7.25, 9))
plt.suptitle("Domain Specific Cognitive Deficits Associated with Age and T2DM\n" \
             "Quantified Separately for Sexes, UK Biobank")

# Panels A & B
# ------

for c, case in enumerate(cases):

    # Panel index
    pi = int(c/2)

    # Dodge index (index within panel)
    di = c%2

    # Current case's dataframe
    df = data[case]

    # Add new line character into x labels
    df["label"] = df["label"].str.replace("_", "\n")
    df["label"] = df["label"].str.replace(" ", "\n")
#    df["label"] = df["label"].str.replace("Short\nTerm\nMemory", "Short-Term\nMemory")

    # Sort labels alphabetically
#    df = df.sort_values(by="label", ignore_index=True)

    # Pick subplot
    plt.subplot(int(len(cases)/2), 1, pi+1)

    # Populate plot


    # Colors
    colors_all = colors_from_values(
        np.array(list(-df["beta"])*cms[c] + [df["beta"].min() + cmo[c], df["beta"].max()*cms[c]]),
        colors[c])

    for i, item in enumerate(df.iterrows()):

        # Extract data
        x = item[0]
        ss, p, y, t, conf_int  = \
            item[1][["sample_sizes", "pval", "beta", "tval", "conf_int"]]

        conf_dist = abs(y - np.array(conf_int))[:, None]

        # Blob for representing value and sample size
        plt.scatter(x=x-(x_dodge-di*2*x_dodge), y=y, s=sum(ss)/sfs[c], color=colors_all[i])
                    #"mediumblue")

        # Plot center of estimate
        plt.scatter(x=x-(x_dodge-di*2*x_dodge), y=y, s=15*lw, color="k")

        # Errorbars
        plt.errorbar(x-(x_dodge-di*2*x_dodge), y, yerr=conf_dist, capsize=4*lw, capthick=0.75*lw,
                      elinewidth=0.75*lw, color="black")

    #    # Annotate stats as text
    #    text = f"T={t:.1f} \n {pformat(p)}" + p2star(p) \
    #        +  f"\n$\mathbf{{N_{{T2DM}}}}$={ss[1]:,}\n$\mathbf{{N_{{ctrl}}}}$={ss[0]:,}"
    #
    #    text_y = 0.2 # 0.5 if max(conf_int) < 0 else max(conf_int) + 0.5
    #    va = "bottom" # "bottom" if y > 0 else "top"
    #    plt.annotate(text, xy=[x, text_y],
    #                 fontsize=9*fs, ha="center", va=va)

        # Add statistical asterisks
        text = p2star(p)
        text_x = x + 0.00 -(x_dodge-di*2*x_dodge)
        if y < 0:
            text_y = min(min(conf_int), y-np.sqrt(sum(ss)/sfs[c])/2e2) - textpads[c]
        else:
            text_y = max(max(conf_int), y+np.sqrt(sum(ss)/sfs[c])/1.5) + textpads[c]

        va = "top" if y < 0 else "bottom"
        plt.annotate(text, xy=[text_x, text_y], fontsize=8*fs,
                     ha="center", va=va, fontweight="bold",
                     rotation=0)


    # Format
    # Add title
    plt.title(titles[pi])

    # Limits
    plt.xlim([-0.5, len(df)-0.5])
    # plt.ylim(ylims[pi])

    # Labels
    plt.ylabel(ylabeltexts[pi])
    if c == len(cases)-1:
        plt.xlabel("\nCognitive domains")

    plt.gca().get_yaxis().set_major_formatter(
            mtc.FuncFormatter(lambda x, p: format(f"{x:.1f}")))

    # Ticks/lines
    plt.axhline(0, linewidth=0.75*lw, color="black", dashes=[4, 4])
    plt.xticks(ticks=np.arange(len(df)), labels=df["label"],
               rotation=xtickrots[c], va=xtickvas[c])
    plt.gca().tick_params(axis="x", pad=xtickpads[c])
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    for sp in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[sp].set_linewidth(.75*lw)
        plt.gca().spines[sp].set_color("black")

    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    # Add scale
    plt.scatter(x=len(df)-sdxo[c], y=ylims[c][0] * 0.88, s=sfscf[c]/sfs[c],
                color="gray")
#    plt.scatter(x=len(df)-sdxo2[c], y=ylims[c][0] * 0.88, s=sfscf2[c]**2/sfs[c],
#                color="lightgray")

    plt.annotate(text=f"Scale:\n{'N' if c<2 else 'K'}={int(sfscf[c])}",
                 xy=[len(df)-sdxo[c], ylims[c][0] * 0.88], va="center", ha="center")


#print(df.groupby("diab")["age"].describe())

# Manual legend
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(mpl.patches.Patch(facecolor="firebrick", edgecolor="k", label='F'))
handles.append(mpl.patches.Patch(facecolor="royalblue", edgecolor="k", label='M'))

# plot the legend
plt.legend(handles=handles, labels=["Females", "Males"], bbox_to_anchor=(0.2, 0.52),
           bbox_transform=plt.gcf().transFigure, frameon=False)


# Save
# ------
plt.tight_layout(h_pad=2)
plt.savefig(OUTDIR + "figures/JAMA_meta_figure_cognition_sex_strat.pdf",
            transparent=True)

# plt.close()
