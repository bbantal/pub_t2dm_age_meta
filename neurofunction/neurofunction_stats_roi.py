#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 00:22:55 2021

@author: botond

Notes:
-this script performs linear regression on functional MRI results (ALFF)
in voxel space

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

from pprint import pprint
from nilearn import image
from nilearn import masking
from nilearn import input_data
from nilearn import plotting
import nibabel as nib

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, check_assumptions
get_ipython().run_line_magic('cd', 'neurofunction')


# =============================================================================
# Setup
# =============================================================================

plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
RLD = True  # Reload regressor matrices instead of computing them again
BATCH = 7  # Batch of preprocessed images to use

T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
excl_sub = [] # [1653701, 3361084, 3828231, 2010790, 2925838, 3846337,]  # Subjects
## to exlucde due to abnormal total gray matter neurofunctions

CTRS = "age"  # Contrast: diab or age
CT = 12  # Cluster threshold for group level analysis (in voxels?)
PTHR = 0.05  # Significance threshold for p values
GM_THR = 0.5  # Threshold for gm mask. Voxels with Probability above this
MDL = "ALFF"  # Modality
EXTRA = ""
PARC = 46  #TODO

VOLUME = 1000  # Cluster volume, mm3
VOXEL_DIM = 2.4  # Voxel dimension in work space, assuming cubic voxels, mm

regressors_dict = {
    "diab": ["subject_label", "diab", "age", "sex", "college", "bmi", "ses"],
    "age": ["subject_label", "age", "sex", "college", "bmi", "ses"],
    "metformin": ["subject_label", "duration", "metformin", "age", "sex",
                  "college", "bmi", "ses"]
    }

print("Settings:\n" \
    f"CONTRAST={CTRS}, MDL={MDL}, PTHR={PTHR}, CT={CT}")

# raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Status
print("Loading data.")

# Load batch
# -------
img_subs = pd.read_csv(SRCDIR + "neurofunction/pub_meta_subjects_batch7_alff.csv",
                       index_col=0)
# Load regressors
# ------
# Age
age = pd.read_csv(SRCDIR + "ivs/age.csv", index_col=0)[["eid", "age-2"]] \
    .rename({"age-2": "age"}, axis=1)

# Sex
sex = pd \
    .read_csv(SRCDIR + "ivs/sex.csv", index_col=0)[["eid", "sex"]] \
    .set_axis(["eid", "sex"], axis=1)

# Diabetes diagnosis
diab = pd.read_csv(SRCDIR + "ivs/diab.csv", index_col=0)[["eid", "diab-2"]] \
    .rename({"diab-2": "diab"}, axis=1) \
    .query('diab >= 0')

# College
college = pd.read_csv(SRCDIR + "ivs/college.csv", index_col=0)[["eid", "college"]] \

# Ses
ses = pd.read_csv(SRCDIR + "ivs/ses.csv", index_col=0)[["eid", "ses"]]

# BMI
bmi = pd.read_csv(SRCDIR + "ivs/bmi.csv", index_col=0)[["eid", "bmi-2"]] \
    .rename({"bmi-2": "bmi"}, axis=1) \
    .dropna(how="any")

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Remove diabetic subjects with missing age of onset OR have too early age of onset
# which would suggest T1DM. If age of onset is below T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# =============================================================================
# Create gm mask
# =============================================================================

# Status
print("Preparing gm mask.")

# Load target affine and target shape (ignore values except for shape and affine)
target_img = image.load_img(HOMEDIR + "tools/utils/target_img_ukb_space.nii.gz")

# Load MNI gm mask
gm_mask_raw = image.load_img(HOMEDIR + "tools/utils/mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Binarize
gm_mask_binarized = image.math_img(f'img > {GM_THR}', img=gm_mask_raw)

# Resample
gm_mask = image.resample_img(
        gm_mask_binarized, target_affine=target_img.affine,
        target_shape=target_img.shape, interpolation="nearest",
        fill_value=0)

# %%
# =============================================================================
# Build regressor matrix
# =============================================================================

# Choose variables
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [diab, age, sex, college, ses, bmi, age_onset]
        ) \
        .drop("age_onset", axis=1)


# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# Inner merge regressors with subjects for whom ALFF has been computed
regressors_clean = regressors.merge(img_subs, on="eid")

# Let's see what matching would look like

# Save full regressor matrix
regressors_clean.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_full_regressors_{CTRS}.csv")

if CTRS == "age":

    # Interactions among independent variables
    var_dict = {
            "sex": "disc",
            "college": "disc",
            "ses": "disc",
            "bmi": "cont"
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1=name,
                var2="age",
                type1=type_,
                type2="cont",
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar_"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="sex",
                vars_to_match=["age"],
                N=1,
                random_state=1
                )

if CTRS == "diab":

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "sex": "disc",
            "college": "disc",
            "ses": "disc",
            "bmi": "cont"
            }

    for name, type_ in var_dict.items():

        check_covariance(
                regressors_clean,
                var1="diab",
                var2=name,
                type1="disc",
                type2=type_,
                save=True,
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar_"
                )

        plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_var="diab",
                vars_to_match=["age", "sex"],
                N=1,
                random_state=1
                )

# Status
print("Covariance checked.")

if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}.csv")

    # Status
    print("Matching performed.")


# =============================================================================
# Analysis
# =============================================================================

# Prep
# ------
# Status
print("Loading regressors.")

# Load regressors
regressors_matched = pd \
    .read_csv(OUTDIR + \
         f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}.csv",
         index_col=0) \
    .rename({"eid": "subject_label"}, axis=1)


# Final sample sizes
print("Total sample size:", regressors_matched.shape[0])
print("Sample sizes:", regressors_matched.groupby([CTRS]).count()["subject_label"])

# Status
print("Loading nifti images.")

# Load input images, careful with the order!
imgs = [image.load_img(SRCDIR + f"{MDL}/{MDL.lower()}_normalized_batch_{BATCH}/" \
                       f"{sub}_{MDL.lower()}_normalized.nii") \
        for sub in regressors_matched["subject_label"].astype(int)]


# Create labeled mask for each region
# ------

# Status
print("Building mask.")

# Load atlas
label_mask = image.load_img(SRCDIR + f"atlas/{PARC}/ukb_gm_labelmask_{PARC}.nii.gz")

# Rshape mask
mask_img = image.resample_img(
        label_mask,
        target_affine=target_img.affine,
        target_shape=target_img.shape,
        interpolation="nearest"
        )

# Save mask img
#nib.save(mask_img, "/Users/botond/Desktop/mask_img.nii")

# Construct masker object (with gm mask also added in)
label_masker = input_data.NiftiLabelsMasker(
        mask_img,
        background_label=0,
        mask_img=gm_mask
        )

labels = pd \
    .read_csv(SRCDIR + f"atlas/{PARC}/ukb_gm_labelmask_{PARC}.csv") \
    ["label"] \
    .str.replace(" ", "_") \
    .to_list()

# Mask ALFF in subject data
# ------
# Assuming that the lsit named "imgs" was not manipulated and has images in the
# same order as "regressors_matched"

# Status
print("Applying masking.")

# Get subjects


# Dictionary to store data
signal_dict = {}

# Iterate through all subjects
for i, subject in tqdm(enumerate(regressors_matched["subject_label"]),
                       total=len(regressors_matched["subject_label"]),
                       desc="Extracting signal from subject: "):

    # Apply masking to ALFF image
    signal = label_masker.fit_transform([imgs[i]])

    # Construct dictionary
    signal_dict[subject] = list(signal[0])

# Status
print("Transforming extracted signal.")

# Transform into dataframe
data = pd \
    .DataFrame(signal_dict.items()) \
    .set_axis(["subject_label", "data"], axis=1) \
    .explode("data") \
    .assign(**{"label": \
               [item for item in labels] \
               *len(regressors_matched["subject_label"])}) \
    .pivot(index="subject_label", columns="label", values="data") \
    .astype(float) \
    .reset_index()

# Build full matrix
df = regressors_matched \
    .merge(data, on="subject_label", how="inner") \
    .rename({"subject_label": "eid"}, axis=1)

# Run stat model
# -----

# Status
print("Fitting models.")

# Dictionary to store stats
feat_stats = {}

# Iterate over all regions
for i, feat in tqdm(enumerate(labels), total=len(labels), desc="Models fitted: "):

    # Prep
    # ----
    # Extract current feature
    sdf = df[["eid", "age", "diab", "sex", "college", "ses", "bmi", f"{feat}"]]

    # Get sample sizes
    sample_sizes = sdf.groupby("diab")["eid"].count()

    # Fit
    # -----
    # Formula
    if CTRS == "age":
        formula = f"{feat} ~ age + C(sex) + C(college) + C(ses) + bmi"
    if CTRS == "diab":
        formula = f"{feat} ~ C(diab) + age + C(sex) + C(college) + C(ses) + bmi"

    # Fit model
    model = smf.ols(formula, data=sdf)
    results = model.fit()

    # Monitor
    # --------

    # Save detailed stats report
    with open(OUTDIR + f"stats_misc/pub_meta_neurofunction_roi_regression_table_{feat}" \
              f"_{CTRS}_{PARC}.html", "w") as f:
        f.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
            results,
            sdf,
            prefix=OUTDIR + \
            f"stats_misc/pub_meta_neurofunction_roi_stats_assumptions_{feat}" \
                "_{CTRS}_{PARC}"
            )

    # Plot across age
    if CTRS == "diab":
        plt.figure()
        plt.title(feat)
        sns.lineplot(data=sdf[[feat, "age", "diab"]], x="age", y=feat, hue="diab",
                     palette=sns.color_palette(["black", "red"]))
        plt.tight_layout()
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_neurofunction_roi_age-" \
                    f"diab-plot_{feat}_{PARC}.pdf")
        plt.close()

    # Save results
    # -------
    # Get relevant key for regressor
    rel_key = [key for key in results.conf_int().index.to_list() \
           if CTRS in key][0]

    # Get effect size
    tval = results.tvalues.loc[rel_key]
    beta = results.params.loc[rel_key]

    # Get confidence intervals
    conf_int = results.conf_int().loc[rel_key, :]

    # Get p value
    pval = results.pvalues.loc[rel_key]

    # Save stats as dict
    feat_stats[f"{feat}"] = [list(sample_sizes), tval, pval, beta,
                              np.array(conf_int)]


# Convert stats to df and correct p values for multicomp
df_out = pd.DataFrame.from_dict(
        feat_stats, orient="index",
        columns=["sample_sizes", "tval", "pval", "beta", "conf_int"]) \
        .reset_index() \
        .rename({"index": "label"}, axis=1) \
        .assign(**{"pval": lambda df: pg.multicomp(list(df["pval"]),
                                                   method="bonf")[1]})

# Save outputs
df_out.to_csv(OUTDIR + f"stats/pub_meta_neurofunction_roi_stats_{CTRS}_{PARC}.csv")

# Status
print("Execution has finished.")