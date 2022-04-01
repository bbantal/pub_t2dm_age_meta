#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:48:01 2022

@author: benett
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
CTRS = "diab"  # Contrast: diab or age
EXTRA = "_sex"
T1DM_CO = 20  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
RLD = 0

print("\nRELOADING REGRESSORS!\n") if RLD else ...

raise

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

# Systolic blood pressure
sbp = pd \
    .read_csv(SRCDIR + "ivs/sbpa.csv", index_col=0) \
    .merge(pd.read_csv(SRCDIR + "ivs/sbpm.csv", index_col=0), on="eid", how="outer") \
    .set_index("eid") \
    [["sbpa_2", "sbpa_2.1", "sbpm_2", "sbpm_2.1"]] \
    .mean(axis=1) \
    .rename("sbp") \
    .reset_index() \
    .dropna()

# Diastolic blood pressure
dbp = pd \
    .read_csv(SRCDIR + "ivs/dbpa.csv", index_col=0) \
    .merge(pd.read_csv(SRCDIR + "ivs/dbpm.csv", index_col=0), on="eid", how="outer") \
    .set_index("eid") \
    [["dbpa_2", "dbpa_2.1", "dbpm_2", "dbpm_2.1"]] \
    .mean(axis=1) \
    .rename("dbp") \
    .reset_index() \
    .dropna()

# Hypertension based on sbp and dbp
htn = pd \
    .merge(sbp, dbp, on="eid", how="inner")\
    .set_index("eid") \
    .eval('sbp >= 140 | dbp >= 90') \
    .astype(int) \
    .rename("htn") \
    .reset_index()

# %%
# Restrict regressors
# -----

# Remove subjects below the age of 40
# age = age.query('age < 50')

# Remove female subjects who did not report menopause
mp = mp \
   .merge(sex, on="eid", how="right") \
   .query('(sex==0 & mp==1) or (sex==1 & mp!=mp)') \
   [["eid", "mp"]]

# Remove female subjects who have reported ongoing hrt at the time
hrt = hrt \
    .merge(sex, on="eid", how="right") \
    .query('(sex==0 & hrt != -11) or (sex==1 & hrt!=hrt)') \
    .pipe(lambda df: df.assign(**{"hrt": 1})) \
   [["eid", "hrt"]]

# Remove diabetic subjects with missing age of onset OR have
# too early age of onset which would suggest T1DM. If age of onset is below
# T1DM_CO, subject is excluded.
age_onset = age_onset \
    .merge(diab, on="eid", how="inner") \
    .query(f'(diab==0 & age_onset!=age_onset) or (diab==1 & age_onset>={T1DM_CO})') \
    [["eid", "age_onset"]]

# %%
# =============================================================================
# Transform
# =============================================================================

# Status
print("Transforming.")


# Merge IVs and put previously defined exclusions into action (through inner merge)
regressors = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, age_onset, mp, hrt, htn]
        ) \
        .drop(["mp", "hrt"], axis=1)

# %%
tdf = regressors.copy()

tdf.groupby(["htn", "diab"]).count()["eid"]
