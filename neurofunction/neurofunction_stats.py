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
from nistats.design_matrix import make_second_level_design_matrix
from nistats.second_level_model import SecondLevelModel
from nistats.reporting import plot_design_matrix
from nistats.thresholding import map_threshold
import nibabel as nib

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, match_cont, \
    check_assumptions
from helpers.data_loader import DataLoader
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'neurofunction')


# =============================================================================
# Setup
# =============================================================================

# plt.style.use("ggplot")

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
RLD = False  # Reload regressor matrices instead of computing them again
BATCH = 8  # Batch of preprocessed images to use

T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects

CTRS = "diab"  # Contrast: diab or age
CT = 12  # Cluster threshold for group level analysis (in voxels?)
PTHR = 0.05  # Significance threshold for p values
GM_THR = 0.5  # Threshold for gm mask. Voxels with Probability above this
MDL = "ALFF"  # Modality
STRAT_SEX = False  # Stratify sex or not #TODO: need to adjust detrending accordinlgy
SEX = 1  # If stratifying per sex, which sex to keep

EXTRA = ""

VOLUME = 1000  # Cluster volume, mm3
VOXEL_DIM = 2.4  # Voxel dimension in work space, assuming cubic voxels, mm

regressors_dict = {
    "diab": ["subject_label", "diab", "age", "sex", "college", "htn"],
    "age": ["subject_label", "age", "sex", "college", "htn"],
    "sex": ["subject_label", "age", "sex", "college", "htn"],
    }

print("Settings:\n" \
    f"CONTRAST={CTRS}, MDL={MDL}, PTHR={PTHR}, CT={CT}, EXTRA={EXTRA}")

#raise

# %%
# =============================================================================
# Load data
# =============================================================================

# Status
print("Loading data.")

# Load batch
# -------
img_subs = pd.read_csv(SRCDIR + f"neurofunction/pub_meta_subjects_batch{BATCH}_alff.csv",
                       index_col=0)
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
        [age, sex, college, diab, mp, hrt, htn, age_onset, duration]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)


# If contrast is age
if CTRS == "age":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# If contrast is sex
if CTRS == "sex":
    # Exclude subjects with T2DM
    regressors = regressors.query("diab == 0")

# Optional: stratify per sex
if STRAT_SEX:
    # Include subjects of one sex only
    regressors = regressors.query(f"sex == {SEX}")

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
            }

    # for name, type_ in var_dict.items():

    #     check_covariance(
    #             regressors_clean,
    #             var1=name,
    #             var2="age",
    #             type1=type_,
    #             type2="cont",
    #             save=True,
    #             prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar"
    #             )

        # plt.close("all")

    if RLD == False:

        # Match
        regressors_matched = match_cont(
                df=regressors_clean,
                main_vars=["age"],
                vars_to_match=["sex", "college", "htn"],
                value=3,
                random_state=11
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
                prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar"
                )

        plt.close("all")

    if RLD == False:

        # Match
        regressors_matched = match(
            df=regressors_clean,
            main_vars=["diab"],
            vars_to_match=["age", "sex", "college", "htn"],
            random_state=11
            )

if CTRS == "sex":

    # Interactions among independent variables
    var_dict = {
            "age": "cont",
            "college": "disc",
            "htn": "disc"
            }

    # for name, type_ in var_dict.items():

    #     check_covariance(
    #             regressors_clean,
    #             var1="sex",
    #             var2=name,
    #             type1="disc",
    #             type2=type_,
    #             save=True,
    #             prefix=OUTDIR + "covariance/pub_meta_neurofunction_covar"
    #             )

    plt.close("all")

    if RLD == False:
        # Match
        regressors_matched = match(
                df=regressors_clean,
                main_vars=["sex"],
                vars_to_match=["age", "college", "htn"],
                random_state=11
                )


# Status
print("Covariance checked.")

if RLD == False:
    # Save matched regressors matrix
    regressors_matched.to_csv(OUTDIR + f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}{EXTRA}.csv")

    # Status
    print("Matching performed.")

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%
# =============================================================================
# Sample sizes
# =============================================================================

if ~STRAT_SEX:


    # CTRS specific settings
    dc = 1 if CTRS == "diab" else 0
    ylim = 300 if CTRS == "diab" else 1500

    # Load regressors
    regressors_matched = pd.read_csv(
            OUTDIR + f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}{EXTRA}.csv"
            )

    # Figure
    plt.figure(figsize=(3.5, 2.25))

    # Plot
    sns.histplot(data=regressors_matched.query(f'diab=={dc}'),
                 x="age", hue="sex",
                 multiple="stack", bins=np.arange(50, 85, 5),
                 palette=["crimson", "royalblue"], alpha=1, zorder=2)

    # Annotate total sample size
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
    plt.title("Brain Activation (ALFF)", fontsize=10)

    # Save
    plt.tight_layout(rect=[0, 0.00, 1, 0.995])
    plt.savefig(OUTDIR + f"stats_misc/pub_meta_neurofunction_sample_sizes_{CTRS}.pdf",
                transparent=True)

    # Close all
    plt.close("all")


# %%
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
         f"regressors/pub_meta_neurofunction_matched_regressors_{CTRS}{EXTRA}.csv",
         index_col=0) \
    .rename({"eid": "subject_label"}, axis=1)


# Final sample sizes
print("Total sample size:", regressors_matched.shape[0])
print("Sample sizes:", regressors_matched.groupby([CTRS]).count()["subject_label"])

# Status
print("Loading nifti images.")

# List of input images, careful with the order!
imgs = [image.load_img(SRCDIR + f"{MDL}/{MDL.lower()}_normalized_batch_{BATCH}/" \
                       f"{sub}_{MDL.lower()}_normalized.nii") \
        for sub in regressors_matched["subject_label"].astype(int)]

# Status
print("Preparing Nistats design matrix.")

# Get subject labels
subject_labels = regressors_matched["subject_label"]

# Make nistats deisgn matrix object
design_matrix = make_second_level_design_matrix(
        subject_labels,
        regressors_matched[regressors_dict[CTRS]]
        )

# Run regression
# -----------

# Status
print("Fitting model.")

# Construct second level GLM object
GLM = SecondLevelModel(mask_img=gm_mask, verbose=2)

# Fit GLM
GLM.fit(imgs, design_matrix=design_matrix)

# Extract results that belong to contrast of interest,
# approximating t scores with z scores due to high N
z_map = GLM.compute_contrast(second_level_contrast=CTRS,
                             output_type="z_score")

# Save Z map
nib.save(z_map, OUTDIR + f"stats/pub_meta_neurofunction_zmap_{MDL.lower()}_" \
                 f"batch{BATCH}_GM_{GM_THR}_contrast_{CTRS}{EXTRA}.nii")


# Threshold voxel maps
# -----------

# Status
print("Thresholding z-map.")

# Apply statistical thresholds
thresholded_map, threshold = map_threshold(
    z_map, alpha=PTHR, height_control='fdr', cluster_threshold=CT)

# Change zeros to nans
thr_map_nan = image.math_img('np.where(img == 0, np.nan, img)', img=thresholded_map)

# Export thresholded image
nib.save(thr_map_nan, OUTDIR + f"stats/pub_meta_neurofunction_statthr_" \
         f"{MDL.lower()}_batch{BATCH}_GM_{GM_THR}_contrast_{CTRS}_uc{CT}_" \
         f"fdr{PTHR}{EXTRA}.nii")


# Status
print("Finished with extracting basic results.")

# <><><><><><><><>
# raise
# <><><><><><><><>

# %%
# =============================================================================
#  Check assumptions in maunally selected clusters
# =============================================================================

# Set style for plotting
from helpers.plotting_style import plot_pars, plot_funcs
lw=1.5

# Create labeled mask for each region
# ------

# Define clusters to look at
clusters_dict = {
        "Caudate_R": [11, 3, 16],
        "Orbitofrontal_Cortex_R": [8, 12, -20],
        "Subgenual_Area_R": [4, 7, -11],
        "Premotor_Cortex_L": [-52, 10, 28],
        "Posterior_Cingulate_Gyrus_L": [-4, -34, 32],
        "Brainstem": [2, -32, -8]
            }

# Status
print("/n/n#####/nChecking assumptions around the following coordinates:")
pprint(clusters_dict)

# Status
print("Building masker object.")

# Function that return radius in number of voxels
comp_radius = lambda volume, voxel_dim: (volume*3/4/3.14159)**(1/3)/voxel_dim

# Transfrom cluster info
clusters = pd \
    .DataFrame(clusters_dict) \
    .T \
    .reset_index() \
    .set_axis(["label", "x", "y", "z"], axis=1) \
    .assign(**{
            "radius": comp_radius(VOLUME, VOXEL_DIM),
            "index": np.arange(len(clusters_dict))+1
            })

# Create clusters in 3d space based on center and radius
# assign labels to clusters
# and then merge them into one array sequentially

# Make voxels within the sphere n, others 0
def create_bin_sphere(arr_size=None, center=None, r=None, value=None):
    """
    Function from SO: https://stackoverflow.com/a/56060957/11411944
    Make 3d sphere
    """
    coords = np.ogrid[:arr_size[0], :arr_size[1], :arr_size[2]]
    distance = np.sqrt((coords[0] - center[0])**2 + \
                       (coords[1]-center[1])**2 + \
                       (coords[2]-center[2])**2)
    return value*(distance <= r)

# MNI to work space coordinate conversion
aff = target_img.affine
mni_convert = lambda mni_coords, aff: \
                    np.linalg.inv(aff[:3, :3])@(mni_coords - aff[:3, 3])

# Get all zero mask to serve as 3d space
space = np.array(target_img.get_data())
space[:] = 0

# Array which will be the mask with labels
label_mask = np.copy(space)

# Do for every cluster
for (i, cluster) in clusters.iterrows():

    # Construct 3d sphere
    current_cluster = create_bin_sphere(
            arr_size=space.shape,
            center=mni_convert(cluster[["x", "y", "z"]], aff),
            r=(cluster["radius"]-0.5),
            value=cluster["index"]
            )

    # Print size
    print(f'size of cluster #{int(cluster["index"])}:',
          np.argwhere(current_cluster == cluster["index"]).shape[0])

    # Add label cluster to mask, sequentially
    label_mask = \
        np.where(current_cluster == cluster["index"],
                 current_cluster,
                 label_mask)

# Conver label_mask to nifti
mask_img = image.new_img_like(
        target_img, data=label_mask
        )
# Save mask img
#nib.save(mask_img, "/Users/botond/Desktop/mask_img.nii")

# Construct masker object (with gm mask also added in)
label_masker = \
    input_data.NiftiLabelsMasker(mask_img, background_label=0, mask_img=gm_mask)


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
                       desc="extracting signal from subject: "):

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
               [item for item in clusters["label"].to_list()] \
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

# Run stats for each label
for label in clusters["label"]:

    # Status
    print(f"Fitting model for {label}.")

    # Formula
    if CTRS == "age":
        formula = f"{label} ~ age + C(sex) + C(college) + htn"
    if CTRS == "diab":
        formula = f"{label} ~ C(diab) + age + C(sex) + C(college) + htn"
    if CTRS == "sex":
        formula = f"{label} ~ age + C(sex) + C(college) + htn"

    # Construct OLS model
    model = smf.ols(formula, data=df)

    # Fit model
    results = model.fit()

    # Save results
    with open(OUTDIR + f"stats_misc/pub_meta_neurofunction_regression_table" \
              f"_{label}_{CTRS}{EXTRA}.html", "w") as file:
        file.write(results.summary().as_html())

    # Check assumptions
    check_assumptions(
        results,
        df,
        prefix=OUTDIR + \
        f"stats_misc/pub_meta_neurofunction_stats_assumptions_{label}_{CTRS}{EXTRA}"
        )

    # Status
    print(f"Visualizing relationships for {label}.")

    # Lineplot across age
    if CTRS == "diab":
        gdf = df \
            [[f"{label}", "age", "diab"]] \
            .pipe(lambda df: df.assign(**{"age_group":
                    pd.cut(df["age"], bins=np.arange(0, 100, 5)).astype(str)
                    })) \
            .sort_values(by="age")

        plt.figure(figsize=(3.5, 3))
        plt.title(f"{label}")
        sns.lineplot(data=gdf, x="age_group", y=f"{label}", hue="diab",
                     palette=sns.color_palette(["black", "red"]),
                     ci=68, err_style="bars", marker="o",
                     linewidth=1*lw, markersize=3 *lw, err_kws={"capsize": 2*lw,
                         "capthick": 1*lw, "elinewidth": 1*lw})
        legend_handles = plt.gca().get_legend().legendHandles
        plt.legend(handles=legend_handles, labels=["HC", "T2DM+"], loc="best",
                   fontsize=8, title="")
        plt.xlabel("Age group")
        plt.ylabel("Brain Activation (ALFF)")
        plt.xticks(rotation=45)
        plt.grid()
        plt.title(label.replace("_", " "))

        plt.tight_layout()
        plt.savefig(OUTDIR + f"stats_misc/pub_meta_neurofunction_age-diab-plot_" \
                    f"{label}{EXTRA}.pdf", transparent=True)
        plt.close()

# Status
print("Finished execution.")
