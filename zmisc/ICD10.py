#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:13:57 2021

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
import matplotlib.pyplot as plt
import matplotlib.ticker as mtc
import seaborn as sns
import pingouin as pg
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from tqdm import tqdm
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.regression_helpers import check_covariance, match, check_assumptions, detrender
from helpers.plotting_style import plot_pars, plot_funcs
get_ipython().run_line_magic('cd', 'volume')

# TODO: for compting age contrast, match for age separately within HC only.

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/volume/"

# Inputs
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate

# %%
# =============================================================================
# Load data
# =============================================================================

# Load volume data
# -------
# Load atrophy data
data = pd.read_csv(SRCDIR + "volume/volume_data.csv").drop(["age", "gender"], axis=1) \
    [["eid", '25005-2.0']].rename({'25005-2.0': "volume"}, axis=1)

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
    .query('diab in [0, 1]')

# College
college = pd.read_csv(SRCDIR + "ivs/college.csv", index_col=0)[["eid", "college"]] \

# BMI
bmi = pd.read_csv(SRCDIR + "ivs/bmi.csv", index_col=0) \
    .rename({"bmi-2": "bmi"}, axis=1) \
    [["eid", "bmi"]] \
    .dropna()

# Menopause
mp = pd \
   .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
   .pipe(lambda df: df.assign(**{
      "mp": df.eval('mp_0 == 1 | mp_1 == 1 | mp_2 == 1').astype(int)})) \

# Hormone replacement therapy
hrt = pd.read_csv(SRCDIR + "ivs/hrt_age.csv", index_col=0) \
    .rename({"hrt_age_2": "hrt"}, axis=1) \
    [["eid", "hrt"]]

# Age of diabetes diagnosis (rough estimate!, averaged)
age_onset = pd \
    .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
    .set_index("eid") \
    .mean(axis=1) \
    .rename("age_onset") \
    .reset_index()

# Restrictive set: Remove female subjects who did not report menopause
mp = mp \
   .merge(sex, on="eid", how="right") \
   .query('(sex==0 & mp==1) or (sex==1 & mp!=mp)') \
   [["eid", "mp"]]

# Restrictive set: Remove female subjects who have reported still taking hrt
hrt = hrt \
    .merge(sex, on="eid", how="right") \
    .query('(sex==0 & hrt != -11) or (sex==1 & hrt!=hrt)') \
    .pipe(lambda df: df.assign(**{"hrt": 1})) \
   [["eid", "hrt"]]

# Restrictive set: Remove diabetic subjects with missing age of onset OR have
# too early age of onset which would suggest T1DM. If age of onset is below
# T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Inspect inpatient diagnoses
# =============================================================================

# Load
# -----

# Primary
dprim = pd.read_csv("~/Desktop/diagn_prim.csv", index_col=0).set_index("eid")

# Secondary
dsec = pd.read_csv("~/Desktop/diagn_sec.csv", index_col=0).set_index("eid")

# Inspect diabetes
# ------

# Look for relevant diagnostic codes:
[val for val in dsec.stack().to_list() if "E11" in val]

# Counts
[(dprim==f"E11{i}").astype(int).sum().sum() for i in np.arange(0, 11, 1)]
[(dsec==f"E11{i}").astype(int).sum().sum() for i in np.arange(0, 11, 1)]

# Reference based on self-report
diab.query("diab==1")

# Inspect cardivascular issues
# ------
[val for val in dprim.stack().to_list() if "I67" in val]

# Uniquue subjects
I_prim_set = set(
    itertools.chain(*[
        (dprim==f"I{i:0>3}").astype(int).sum(axis=1).rename("total") \
            .to_frame().query('total > 0').index.to_list() \
                for i in np.arange(0, 1000, 1)
                    ])
            )

I_sec_set = set(
    itertools.chain(*[
        (dsec==f"I{i:0>3}").astype(int).sum(axis=1).rename("total") \
            .to_frame().query('total > 0').index.to_list() \
                for i in np.arange(0, 1000, 1)
                    ])
            )

# Combine prim and sec
I_combined_set = {*I_prim_set, *I_sec_set}

# Counts
len(I_prim_set)
len(I_sec_set)
len(I_combined_set)

# Compare to availability
pd.DataFrame(I_prim_set, columns=["eid"]).sort_values(by="eid", ignore_index=True).merge(diab, on="eid", how="inner").query('diab==1')
pd.DataFrame(I_sec_set, columns=["eid"]).sort_values(by="eid", ignore_index=True).merge(diab, on="eid", how="inner").query('diab==1')
pd.DataFrame(I_combined_set, columns=["eid"]).sort_values(by="eid", ignore_index=True).merge(diab, on="eid", how="inner").query('diab==1')

# Reference
diab.query("diab==1")
