#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:49:07 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'med')
get_ipython().run_line_magic('matplotlib', 'inline')

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/med/cognition/"

# Inputs
CTRS = "metfonly_unmed"
CTRS_label = "Medicated (T2DM+, metformin only) vs. Unmedicated (T2DM+)"

# Case specific values
cases = [CTRS]
titles = [
        "Domain Specific Cognitive Performance\n" \
        f"{CTRS_label}"
            ]
ylabeltexts = [
        f"Percentage difference\nin cognitive performance\nmedicated vs " \
            "unmedicated (% of avg)",
        ]
colors = ["PRGn"]
ylims = [[-40, 15]]
sfs = [5e-1]  # Marker size factors
sfscf = [500]  # Marker size scale factors
sdxo = [0.8]  # x axis offset of scale info
textpads = [0.1]  # Padding for text along y axis
xtickrots = [0]  # Rotation of xticks
xtickvas = ["top"]  # Vertical alignment for xticks
xtickpads = [0]  # Paddong of xticks

# Custom label order for plot
label_order = [
        "Executive\nFunction",
        "Processing\nSpeed",
        "Reaction\nTime",
        "Numeric\nMemory",
        "Abstract\nReasoning"
        ]

#raise

# %%

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Load data
# ------

# Dict to store data
data = {}

# Iterate over all cases (age, diab, meta...)
for case in cases:

    # Load
    df = pd.read_csv(
            OUTDIR + f"stats/pub_meta_med_cognition_stats_{case}.csv",
            index_col=0
            )

    # Cast strings to float arrays
    df["sample_sizes"] = \
        df["sample_sizes"].apply(lambda item: np.array(item[1:-1].split(", "), dtype=float))
    df["conf_int"] = \
        df["conf_int"].apply(lambda item:
            np.array([float(val) for val in item[1:-1].split(" ") if len(val) > 0])
        )

    # Assign transformed df to data dict
    data[case] = df

# =============================================================================
# Figure
# =============================================================================

f = plt.figure(figsize=(7.25, 3))
plt.suptitle(titles[0])

# Panels A & B
# ------

for c, case in enumerate(cases):

    # Current case's dataframe
    df = data[case]

    # Add new line character into x labels
    df["label"] = df["label"].str.replace("_", "\n")
    df["label"] = df["label"].str.replace(" ", "\n")
    df["label"] = df["label"].str.replace("Short\nTerm\nMemory", "Short-Term\nMemory")

    # Sort labels alphabetically
    df = df.sort_values(by="label", ignore_index=True)

    # Pick custom order
    df = df.set_index("label").loc[label_order].reset_index()

    # Pick subplot
    plt.subplot(len(cases), 1, c+1)

    # Populate plot
    # Colors
    colors_all = colors_from_values(
        df["beta"], colors[c], vmin=min(df["beta"])/3, vmax=-min(df["beta"])/3)

    for i, item in enumerate(df.iterrows()):

        # Extract data
        x = item[0]
        ss, p, y, t, conf_int  = \
            item[1][["sample_sizes", "pval", "beta", "tval", "conf_int"]]

        conf_dist = abs(y - np.array(conf_int))[:, None]

        # Blob for representing value and sample size
        plt.scatter(x=x, y=y, s=sum(ss)/sfs[c], color=colors_all[i])
                    #"mediumblue")

        # Plot center of estimate
        plt.scatter(x=x, y=y, s=15*lw, color="k")

        # Errorbars
        plt.errorbar(x, y, yerr=conf_dist, capsize=4*lw, capthick=0.75*lw,
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
        text_x = x + 0.00
        text_y =  min(conf_int) - textpads[c] \
                if y < 0 else max(conf_int) + textpads[c]
        va = "top" if y < 0 else "bottom"
        plt.annotate(text, xy=[text_x, text_y], fontsize=12*fs,
                     ha="center", va=va, fontweight="bold",
                     rotation=0)

    # Format
    # Add title
    # ttl = plt.title(titles[c])
#    ttl.set_x(ttl.get_position()[0])

    # Limits
    plt.xlim([-1, len(df)-0.5])
    plt.ylim(ylims[c])

    # Labels
    plt.ylabel(ylabeltexts[c])
    if c == len(cases)-1:
        plt.xlabel("\nCognitive domains")

    plt.gca().get_yaxis().set_major_formatter(
            mtc.FuncFormatter(lambda x, p: format(f"{x:.1f}")))

    # Ticks/lines
    plt.axhline(0, linewidth=lw*0.5, color="black", dashes=[4, 4])
    plt.xticks(ticks=np.arange(len(df)), labels=df["label"],
               rotation=xtickrots[c], va=xtickvas[c])
    plt.gca().tick_params(axis="x", pad=xtickpads[c])
    plt.gca().xaxis.tick_bottom()
    plt.gca().yaxis.tick_left()

    for sp in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[sp].set_linewidth(0.75*lw)
        plt.gca().spines[sp].set_color("black")

    plt.gca().xaxis.grid(False)
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)

    # Add arrow representing directionality
    plt.gca().annotate("",
                xy=(0.1, 0.95), xycoords='axes fraction',
                xytext=(0.1, 0.1), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="fancy, head_width=1, head_length=2, tail_width=0.8",
                                connectionstyle="arc3",
                                facecolor='lightgrey',
                                linewidth=1),
                va="center", ha="center", fontsize=9, rotation=90
                )
    plt.gca().annotate("Improvement\nwith medication",
                xy=(0.1, 0.8), xycoords='axes fraction',
                xytext=(0.065, 0.42), textcoords='axes fraction',
                va="center", ha="center", fontsize=9, rotation=90
                )

    # Add scale
    plt.scatter(x=len(df)-sdxo[c], y=ylims[c][0] * 0.88, s=sfscf[c]/sfs[c],
                color="gray")

    plt.annotate(text=f"Scale:\nN={sfscf[c]}",
                 xy=[len(df)-sdxo[c], ylims[c][0] * 0.88], va="center", ha="center")

## Caption info, !: not quite correct, cases with scores are more complex than
## just > 0. FOr some it's >=0, and others it's >0
## ---------
#
#stn = list(df.columns).index("f4282")
#id_vars = df.columns[:stn]
#
## Total sample size
#
## Gross total sample size and age
#print(
#  df \
#    .melt(id_vars=id_vars, var_name="feat", value_name="score") \
#    .query(f'feat not in {excl}') \
#    .dropna(subset=["score"], axis=0) \
#    .query('score > 0') \
#    .fillna(value={"age_onset": "NAN"}, axis=0) \
#    .pivot_table(
#            index=['eid', 'age', 'college', 'diab', 'age_onset'],
#            columns="feat") \
#    .reset_index() \
#    .groupby("diab") \
#    ["age"] \
#    .describe()
#    )
#
## Sample sizes and Age per cognitive task
#print(df \
#    .melt(id_vars=id_vars, var_name="feat", value_name="score") \
#    .query(f'feat not in {excl}') \
#    .dropna(subset=["score"], axis=0) \
#    .query('score > 0') \
#    .rename({"feat": "index"}, axis=1) \
#    .merge(labels, on="index") \
#    .groupby(["label", "diab"])["age"] \
#    .describe())
#
## Beta coefficients
#print(df[["label", "beta", "tval", "pval"]])

#print(df.groupby("diab")["age"].describe())

# Save
# ------
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(OUTDIR + f"figures/JAMA_meta_figure_med_cognition_{CTRS}.pdf",
            transparent=True)

plt.close()
