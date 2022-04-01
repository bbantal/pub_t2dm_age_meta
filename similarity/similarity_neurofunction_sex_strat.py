#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:57:25 2021

@author: botond

The goal is to quantify similarity along the following factors:

T2DM - aging
F - M
(Neurofunction only)


"""

import os
import itertools
import functools
import numpy as np
import pandas as pd
from nilearn import image, input_data, masking
import nibabel as nib
from scipy import stats, special
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from adjustText import adjust_text
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'similarity')
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "results/"
OUTDIR = HOMEDIR + "results/similarity/"

PC = 46  # Parcellation version (46 or 139)
gm_thr = 0.5  # GM threshold
CORRMET = "pearson"
EXTRA = "_mean"

contrasts = ["age", "diab"]
strats = ["F", "M"]
modalities = ["neurofunction"]
vol_field = "beta"  # Numerical field to consider from neurofunction data
ds_extra = "alff_batch8_GM_0.5"

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%
# =============================================================================
# Import data
# =============================================================================
"""
Going to put everything into a flat list
"""

# Status
print("Loading data.")

# Lists to store data
data_list = {}  # Final list for deriving similarities
raw_list = {}  # Temp list for storing raw functional images

# Import UKB functional
# -----

# Load the ones that exist
for ct, st in itertools.product(contrasts, strats):

    print(ct, st)

    filename = f"pub_meta_neurofunction_zmap_{ds_extra}_contrast_{ct}_{st}.nii"
    try:
        img = image.load_img(SRCDIR + f"neurofunction/stats/" + filename)
        raw_list[f"neurofunction_{ct}_{st}"] = img
        print(f"Loaded {filename}")
    except Exception as e:
        pass
#        print(f"Did not load {filename}", e)

# Status
print("Loading parcellation.")

# Load anatomical atlas
# -----
# Label mask
label_mask = image.load_img(HOMEDIR + f"data/atlas/{PC}/ukb_gm_labelmask_{PC}.nii.gz")

# Label list
label_list = pd.read_csv(HOMEDIR + f"data/atlas/{PC}/ukb_gm_labelmask_{PC}.csv") \
    .pipe(lambda df:
        df.assign(**{
                "label": df["label"].apply(
                        lambda item: item.replace(" ", "_")
                        )})
    )

# Load gray matter mask
# ---------
# Load mask
gm_img = image.load_img(HOMEDIR + "data/atlas/" \
                        "mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Resample gm_mask
gm_img_rs = image.resample_img(
            gm_img,
            target_affine=label_mask.affine,
            target_shape=label_mask.shape,
            interpolation="nearest"
                )

# Binarize gm mask
gm_mask = image.math_img(f'img > {gm_thr}', img=gm_img_rs)

# %%
# =============================================================================
# Convert functional data
# =============================================================================

# Build mask
# -----

# Status
print("Building mask.")

# Dictionary to store masks
label_masks = {}

# Iterate over all regions
for item in tqdm(list(label_list.itertuples())):

    # Extract ROI's label and value form current row
    label, value = item.label, item.value

    # Build mask
    label_masks[value] = image.math_img(
                                f"np.where(img == {value}, 1, 0)",
                                img=label_mask,
                                        )

# %%
# Apply masks and coarsen
# ------

# Status
print("Applying masks.")

def coarsen_effect(file):
    """
    Quantifies relevance of effects to each anatomical label.
    """

    # Get img file
    img = raw_list[file]

    # Resample img
    img = image.resample_img(
            img,
            target_affine=label_mask.affine,
            target_shape=label_mask.shape
            )

    # Apply gm mask if needed (UKB is already masked)
    if "neuroquery" in file:
        img = image.threshold_img(img, threshold="0%", mask_img=gm_mask)

    # List of coverages for current img
    df_list = []

    # Iterate over all regions
    for item in tqdm(list(label_list.itertuples()), desc=file):

        # Extract ROI's label and value form current row
        label, value = item.label, item.value

        # Mask
        mask_img = label_masks[value]

        # Apply mask
        masker = input_data.NiftiMasker(mask_img)
        masked_img = masker.fit_transform(img)

        # Apply operation
        # A: mean
        # =======
        res = np.mean(masked_img)

#        # B: max/min
#        # =======
#        # Take extremes both ways
#        val_max = np.max(masked_img)
#        val_min = np.min(masked_img)
#
#        # Pick the most extreme by absolute value
#        res = [val_max, val_min][np.argmax(np.abs([val_max, val_min]))]

        # ========
        # Store
        df_list.append(pd.DataFrame([label, res]) \
                       .T.set_axis(["label", "value"], axis=1) \
                       .astype({"value": float}))


    df = pd.concat(df_list, axis=0)

    return df

# Iterate over all raw functional files
for file in raw_list.keys():

    # Status
#    print(file)

    # Coarsen
    data_list[file] = coarsen_effect(file)

# %%
# Merge dataframes
# -----

# Status
print("Merging Dataframes.")

keys = list(data_list.keys())

# Custom order
keys = [
     'neurofunction_age_F',
     'neurofunction_age_M',
     'neurofunction_diab_F',
     'neurofunction_diab_M',
     ]

# Helper functions
unique_col = lambda key: data_list[key].rename({"value": key}, axis=1)
merge = lambda a, b: a.merge(b, on=["label"], how="inner")

# Merge dfs
df = functools.reduce(merge, list(map(unique_col, keys)))

# Reformat column names
df.columns = list(map(lambda x: x.replace("_", " "), list(df.columns)))

# Save for later
#df.to_csv(OUTDIR + f"df_{PC}{EXTRA}.csv")
#df.info()

# Open
#df = pd.read_csv(OUTDIR + f"df_.csv")

# %%
# =============================================================================
# Derive similarities
# =============================================================================

# Compute correlations
#corr_matrix = df.set_index(["index", "label"]).corr(method=CORRMET)

# Compute staistical significance of correlations
def comp_corr_pvals(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    corr_matrix = dfcols.transpose().join(dfcols, how='outer')
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            corr_matrix[r][c], pvalues[r][c] = eval(f"stats.{CORRMET}r(df[r], df[c])")
    return [corr_matrix.astype(float), pvalues.astype(float)]

corr_matrix, corr_pvals_raw = comp_corr_pvals(df)
# Not bonferroni corrected yet! Correction takes place below!

# Perform Bonferroni correction
# n=number of elements in one of the triangles
corr_pvals = corr_pvals_raw * special.comb(corr_pvals_raw.shape[0], 2)

# %%
# =============================================================================
# Visualize
# =============================================================================

# Unpack plotting utils
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

# Annotation
rhos = corr_matrix.values.flatten()
pvals = corr_pvals.values.flatten()
annot_text = [f"{rhos[i]:.2f}\n" + p2star(pvals[i]) for i in range(len(rhos))]
annot_df = pd.DataFrame(np.array(annot_text).reshape(corr_matrix.shape))

# Correlation plot
plt.figure(figsize=(4.8, 4.2))
#plt.rcParams['xtick.labelsize']=16
#plt.rcParams['ytick.labelsize']=16
plt.title(f"Correlation Based Similarities between\nBrain Activation (ALFF) " \
          "Effects Associated with\nAge and T2DM, Quantified Separately for Sexes",
          )
g = sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap="seismic", annot=annot_df,
            fmt="", linewidth=1, linecolor="k",
            annot_kws={"fontsize": 8*fs})
g.figure.axes[-1].tick_params(labelsize=6*fs)
plt.xticks(rotation=45, ha="right");
plt.tight_layout()
plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_neurofunction_strat{EXTRA}.pdf")



# TODO
## Tidy ip the figure
## --------
# plt.title("")
# plt.xlabel(ticktext)
#
# plt.savefig(OUTDIR + f"figures/corr_matrix_{PC}_{CORRMET}{EXTRA}.pdf")

# %%
# Scatterplot
'''
'ukb_neurofunction_age_F'
'ukb_neurofunction_age_M'
'ukb_neurofunction_diab_F'
'ukb_neurofunction_diab_M'
'''

df.columns = list(map(lambda x: x.replace(" ", "_"), list(df.columns)))

combos = list(itertools.combinations(keys, 2))
icombos = list(itertools.combinations(np.arange(df.shape[1]-1), 2))

# It over
for c, combo in tqdm(enumerate(combos)):

    # Unpack label combo
    a, b = combo

    # Unpack corresponding indexes
    i1, i2 = icombos[c]

    # Seaborn lmplot
    sns.lmplot(data=df, x=a, y=b, height=3.5,
               line_kws={"linewidth": 2, "zorder": 2},
               scatter_kws={"s": 15, "linewidth": 0.7, "edgecolor": "k", "zorder": 3})

    # Optional: annotate labels to points >
    # -----
    # Decide which labels to annotate based on coords, to avoid overlapping text
    # Based on x coord
    # limits_x = np.percentile(np.linspace(df.loc[:, a].min(), df.loc[:, a].max(), 101), [2, 98])
    limits_x = np.percentile(df.loc[:, a], [1, 98])

    # Based on y coord
    # limits_y = np.percentile(np.linspace(df.loc[:, b].min(), df.loc[:, b].max(), 101), [2, 98])
    limits_y = np.percentile(df.loc[:, b], [1, 98])

    # Based on how much it drives the correlation
    pearson_terms = (df.loc[:, a] - df.loc[:, a].mean()) * (df.loc[:, b] - df.loc[:, b].mean())
    # limits_p = np.percentile(np.linspace(pearson_terms.min(), pearson_terms.max()), [10, 80])
    limits_p = np.percentile(pearson_terms, [5, 95])

    cond_x = (df.loc[:, a] < limits_x[0]) | (df.loc[:, a] > limits_x[1])
    cond_y = (df.loc[:, b] < limits_y[0]) | (df.loc[:, b] > limits_y[1])
    # cond_p = (pearson_terms < limits_p[0]) | (pearson_terms > limits_p[1])

    # Final mask
    annot = cond_x | cond_y #| cond_p

    # List of annotations for later adjustment
    annot_list = []

    # Annotate labels
    for i in range(len(df)):
        if annot[i]:
            annot_list.append(
                            plt.annotate(
                                df.loc[i, "label"].replace("_", " "),
                                xy=[df.loc[i, a], df.loc[i, b]],
                                        )
                            )

    # Adjust labels to avoid overlap
    adjust_text(annot_list, arrowprops=dict(arrowstyle="-", linewidth=1,
                                            connectionstyle="angle3"),
                expand_points=[2, 3.5])

    # -----

    # Annotate stats to fig
    text = f"r = {corr_matrix.iloc[i1, i2]:.2f}\n{pformat(corr_pvals.iloc[i1, i2])}"
    plt.annotate(text, xy=[0.05, 0.9], xycoords="axes fraction",
                 bbox=dict(boxstyle='square', fc='white'))

    # Add dashed lines at 0, 0
    plt.axhline(0, linestyle="--", lw=1, color="gray", zorder=1)
    plt.axvline(0, linestyle="--", lw=1, color="gray", zorder=1)

    # Formatting
    plt.xlabel(a.replace("_", " @").replace("diab", "T2DM") \
               .replace("@M", "(Males)").replace("@F", "(Females)") \
               .replace("neurofunction", "Brain Activation (ALFF)") \
                   + "\n(Beta coefficient)")
    plt.ylabel(b.replace("_", " @").replace("diab", "T2DM") \
               .replace("@M", "(Males)").replace("@F", "(Females)") \
               .replace("neurofunction", "Brain Activation (ALFF)") \
                   + "\n(Beta coefficient)")

    xlim = np.array(plt.gca().get_xlim())
    xpad = 0.05*(xlim[1] - xlim[0])
    plt.xlim(xlim[0] - xpad, xlim[1] + xpad)

    plt.tight_layout(rect=[0, 0, 0.96, 1])
    plt.savefig(OUTDIR + f"scatterplots/scatter_{PC}_strat{EXTRA}_{a}_{b}.pdf")
    plt.close("all")

plt.close("all")

# %% Testing ground
#temp = nib.load(WORKDIR + "age.nii.gz")
#nib.save(temp, WORKDIR + "neuroquery_mix_age.nii")
#
#test = image.threshold_img(img, threshold=-999999, mask_img=gm_mask)
#
##test = image.math_img(f'img*
#
#plotting.view_img(test, bg_img=False).open_in_browser()

#mylist = []
#for file, img in raw_list.items():
#    # Resample img
#    img = image.resample_img(
#            img,
#            target_affine=label_mask.affine,
#            target_shape=label_mask.shape
#            )
#
#    # Apply gm mask if needed (UKB is already masked)
#    dat = masking.apply_mask(img, gm_mask)
#
#    mylist.append(pd.DataFrame(dat.flatten(), columns=[file]))
#
#tempdf = pd.concat(mylist, axis=1)
#tempcorr = tempdf.corr(method="spearman")
#
#plt.hist(tempdf.iloc[:, 0])
