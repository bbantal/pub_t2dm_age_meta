 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 23:55:51 2021

@author: botond
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

plt.style.use("default")
#plt.style.use("ggplot")
#sns.set_style("whitegrid")

fs=1.2  # Fontsize
lw=2.0   # Linewidth

plot_pars = [fs, lw]

# Stylesheet
plt.rcParams['xtick.color'] = "black"
plt.rcParams['ytick.color'] = "black"
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 0.5*lw
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 0.5*lw
plt.rcParams['xtick.labelsize']=8*fs
plt.rcParams['ytick.labelsize']=8*fs
plt.rcParams['text.color'] = "black"
plt.rcParams['axes.labelcolor'] = "black"
plt.rcParams["font.weight"] = "regular"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8*fs
plt.rcParams['axes.labelsize']=9*fs
plt.rcParams['axes.labelweight'] = "regular"
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 9*fs
plt.rcParams['legend.title_fontsize'] = 9*fs
# plt.rcParams['text.latex.preamble'] = r'\math'
plt.rcParams['figure.titlesize'] = 10*fs
plt.rcParams['figure.titleweight'] = "regular"
plt.rcParams['axes.titlesize'] = 9*fs
plt.rcParams['axes.titleweight'] = "regular"
#plt.rcParams['axes.axisbelow'] = True

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
#def colors_from_values(values, palette_name):
#    # normalize the values to range [0, 1]
#    normalized = (values - min(values)) / (max(values) - min(values))
#    # convert to indices
#    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
#    # use the indices to get the colors
#    palette = sns.color_palette(palette_name, len(values))
#
#    return np.array(palette).take(indices, axis=0)

def colors_from_values(values, palette_name, vmin=None, vmax=None):
    """ Function to build colormaps from lists """
    # Get boundaries
    if (vmin==None) | (vmax==None):
        vmin = min(values)
        vmax = max(values)

    # Build cmapper object
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap(palette_name))

    # Return color values
    return mapper.to_rgba(values)


# Function to convert float p values to str
def float_to_sig_digit_str(x, k):
    """
    Converts float to string with one significant figure
    while refraining from scientific notation

    inputs:
        x: input float to be converted to string (float)
        k: number of significant figures to keep (int)
    """

    import numpy as np

    # Get decimal exponent of input float
    exp = int(f"{x:e}".split("e")[1])

    # Get rid of all digits but the first figure
    x_fsf = round(x*10**-exp, k-1) * 10**exp

    # Get rid of scientific notation and convert to string
    x_str = np.format_float_positional(x_fsf)

    # Return string output
    return x_str

# Add p values and sample sizes to plot
def pformat(p):
    """ Formats p values for plotting """

#    if p < 0.001:
#        return "$\it{P}$=" + float_to_sig_digit_str(p, 1)
#    elif p > 0.995:
#        return "$\it{P}$=1.0"
#    else:
#        return "$\it{P}$=" + f"{p:.2g}"

    if p > 0.995:
        return "$\it{p}$=1.00"
    elif p >= 0.01:
        return "$\it{p}$=" + f"{p:.2f}" # [1:]
    elif p >= 0.001:
        return "$\it{p}$=" + f"{p:.3f}" # [1:]
    elif p < 0.001:
        return "$\it{p}$<0.001"
    else:
        "INVALID!"

plot_funcs = [p2star, colors_from_values, float_to_sig_digit_str, pformat]
