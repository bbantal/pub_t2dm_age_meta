#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:46:09 2021

@author: botond

if 91*109*91 then voxel size is 2mm (standard MNI space)

"""

# =============================================================================
# Setup
# =============================================================================

import os
from functools import reduce
from nilearn import plotting, image, datasets, masking, glm
import nibabel as nib
# from nistats.thresholding import map_threshold
from matplotlib import pyplot as plt
import matplotlib as mpl
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'neurofunction')
get_ipython().run_line_magic('matplotlib', 'inline')

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/neurofunction/"

# Inputs
MDL = "ALFF"
PTHR = 0.05
BATCH = 8
UC = 12
GM_THR = 0.5  # GM threshold
cut_coords = (-2, 11, 16)

#raise

# Helper functions
# -----
# Look at extremes
def extremes(imgs):
    for img in imgs:
      print(img.get_fdata().min())
      print(img.get_fdata().max())

# Increase resolution
new_affine = lambda img, zoom: \
    nib.affines.rescale_affine(
            img.affine,
            img.shape,
            zoom,
            list(map(lambda x: x*zoom, img.shape))
            )

resample = lambda img, zoom: \
    image.resample_img(
            img,
            target_affine=new_affine(img, zoom),
            target_shape=
                list(map(lambda x: x*zoom, img.shape))
                )

# Load gray matter mask
# ---------
# Load mask
gm_img = image.load_img(HOMEDIR + "data/atlas/" \
                        "mni_icbm152_gm_tal_nlin_asym_09c.nii")

# MNI props
mni = datasets.load_mni152_template()

# Resample gm_mask
gm_img_rs = image.resample_img(
            gm_img,
            target_affine=mni.affine,
            target_shape=mni.shape,
            interpolation="nearest"
                )

# Binarize gm mask
gm_mask = image.math_img(f'img > {GM_THR}', img=gm_img_rs)

# Plotting prep
# -----
fs, lw = plot_pars
p2star, colors_from_values, float_to_sig_digit_str, pformat = plot_funcs

#raise

# %%
# =============================================================================
# Females
# =============================================================================

suffix1 = f"batch{BATCH}_GM_{GM_THR}_contrast_age_F"
suffix2 = f"batch{BATCH}_GM_{GM_THR}_contrast_diab_F"
cmap = 'PiYG_r'
vm = 3

# Load images
back = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix1}.nii')
fore = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix2}.nii')

# (ALready masked and in MNI space)

# Threshold fore image
fore, _ = glm.threshold_stats_img(
   fore, alpha=PTHR, height_control='fdr', cluster_threshold=UC)

#back1 = resample(back1, 4)
#fore1 = resample(fore1, 4)

# Separate pos/neg values
fore_view = fore.get_fdata()
fore_view[fore_view != fore_view]=0
fore_pos = image.math_img('img > 0', img=fore)
fore_neg = image.math_img('img < 0', img=fore)

# Display
fig = plt.figure(figsize=(7.25, 2.2))
#plt.suptitle("Overlaid effects in ALFF with respect to age and T2DM: " \
#             "UK Biobank dataset")

display = plotting.plot_roi(back, cmap=cmap, cut_coords=cut_coords,
                            vmin=-vm, vmax=vm, black_bg=True, figure=fig,
                            colorbar=True, draw_cross=False, annotate=False)
display.add_contours(fore_pos, colors="red", alpha=1, linewidths=0.5*lw)
display.add_contours(fore_neg, colors="blue", alpha=1, linewidths=0.5*lw)

display.annotate(size=7*fs)
display._colorbar_ax.tick_params(labelsize=7*fs)

plt.savefig(OUTDIR + f'figures/JAMA_meta_neurofunction_{MDL}_age-diab_overlap_F.pdf',
            transparent=True)

#plotting.view_img(fore).open_in_browser()

# %%
# =============================================================================
# Males
# =============================================================================

suffix1 = f"batch{BATCH}_GM_{GM_THR}_contrast_age_M"
suffix2 = f"batch{BATCH}_GM_{GM_THR}_contrast_diab_M"
cmap = 'PiYG_r'
vm = 3

# Load images
back = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix1}.nii')
fore = image.load_img(OUTDIR \
               + f'stats/pub_meta_neurofunction_zmap_{MDL}_{suffix2}.nii')

# (ALready masked and in MNI space)

# Threshold fore image
fore, _ = glm.threshold_stats_img(
   fore, alpha=PTHR, height_control='fdr', cluster_threshold=UC)

#back1 = resample(back1, 4)
#fore1 = resample(fore1, 4)

# Separate pos/neg values
fore_view = fore.get_fdata()
fore_view[fore_view != fore_view]=0
fore_pos = image.math_img('img > 0', img=fore)
fore_neg = image.math_img('img < 0', img=fore)

# Display
fig = plt.figure(figsize=(7.25, 2.2))
#plt.suptitle("Overlaid effects in ALFF with respect to age and T2DM: " \
#             "UK Biobank dataset")

display = plotting.plot_roi(back, cmap=cmap, cut_coords=cut_coords,
                            vmin=-vm, vmax=vm, black_bg=True, figure=fig,
                            colorbar=True, draw_cross=False, annotate=False)
display.add_contours(fore_pos, colors="red", alpha=1, linewidths=0.5*lw)
display.add_contours(fore_neg, colors="blue", alpha=1, linewidths=0.5*lw)

display.annotate(size=7*fs)
display._colorbar_ax.tick_params(labelsize=7*fs)

plt.savefig(OUTDIR + f'figures/JAMA_meta_neurofunction_{MDL}_age-diab_overlap_M.pdf',
            transparent=True)

#plotting.view_img(fore).open_in_browser()


# plt.close("all")

