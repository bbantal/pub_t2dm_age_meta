#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:38:41 2020

@author: botond
"""

import os
import numpy as np
import pandas as pd
import itertools
import functools
from IPython import get_ipython

get_ipython().run_line_magic('cd', '..')
from helpers.data_loader import DataLoader
get_ipython().run_line_magic('cd', 'zmisc')


# =============================================================================
# Setup
# =============================================================================
# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/"
OUTDIR = HOMEDIR + "results/"

T1DM_CO = 40  # Cutoff age value for age of diagnosis of diabetes to separate
# T1DM from T2DM. Explained below in more details.
AGE_CO = 50  # Age cutoff (related to T1DM_CO) to avoid T2DM low duration subjects

# raise
# %%
# =============================================================================
# Load data
# =============================================================================

# Initiate loader object
dl = DataLoader()

# Load data
dl.load_basic_vars(SRCDIR)

# Extract relevant variables
age, sex, diab, college, bmi, mp, hrt, age_onset, duration, htn = \
    (dl.age, dl.sex, dl.diab, dl.college, dl.bmi, dl.mp, dl.hrt, dl.age_onset, \
    dl.duration, dl.htn)

# Perform filtering
dl.filter_vars(AGE_CO, T1DM_CO)

# Extract filtered series
age, mp, hrt, age_onset = dl.age, dl.mp, dl.hrt, dl.age_onset

# Merge IVs and put previously defined exclusions into action (through merge)
df = functools.reduce(
        lambda left, right: pd.merge(left, right, on="eid", how="inner"),
        [age, sex, college, diab, mp, hrt, htn, age_onset, duration]
        ) \
        .drop(["mp", "hrt", "age_onset"], axis=1)

# Load medication specific data
CTRS = "metfonly_unmed"  # Contrast: diab or age
med = pd.read_csv(SRCDIR + f"med/{CTRS}.csv")[["eid", CTRS]]
df_med = df.merge(med, on="eid")


# %%
# =============================================================================
# Run queries
# =============================================================================

# Total N
df.shape[0]

# Age
df["age"].describe()

# Females
df.query("sex == 0").shape[0]

# T2DM
df.query("diab == 1").shape[0]

# Age (T2DM)
df.query("diab == 1")["age"].describe()

# Control
df.query("diab == 0").shape[0]

# Age (Control)
df.query("diab == 0")["age"].describe()

# Duration
df["duration"].describe()

# Unmedicated vs Metformin alone
df_med.groupby(CTRS)["eid"].count()

# Other

# df_med.query("metfonly_unmed == 1")["duration"].describe()
# df.query("diab==1")
# df.groupby(["htn", "diab"]).count()
# age.quantile(q=0.01)
# age.describe()


# %%
# Cognition specific sample sizes
# -----

# Load cognitive data
# -------
# Labels
labels = {
     "4282-2.0": "Short_Term_Memory",
     "6350-2.0": "Executive_Function",
     "20016-2.0": "Abstract_Reasoning",
     "20023-2.0": "Reaction_Time",
     "23324-2.0": "Processing_Speed",
     }

# Load data
data = pd \
    .read_csv(SRCDIR + "cognition/cognition_data.csv") \
    [["eid",
      *labels.keys()
     ]]

# Merge with extracted regressors
cogn_merged = df.merge(data, on="eid").set_index(["eid", 'age', 'sex', 'college', 'diab', 'htn', 'duration'])

# Check for how many subjects there were data for at least 1 cognitive task
cogn_keep = cogn_merged.dropna(how="all")

# Reset indexes for queries
cogn_df = cogn_keep.reset_index()

# T2DM
cogn_df.query("diab==1")

# HC
cogn_df.query("diab==0")


# Method B: just the matched samples
# # Combinations
# contrasts = ["age", "diab"]
# features = ["Executive_Function", "Processing_Speed", "Reaction_Time", "Short_Term_Memory", "Abstract_Reasoning"]
# combos = list(itertools.product(contrasts, features))

# # Collection
# df_cogn_list = []

# # It
# for c, combo in enumerate(combos):

#     # Unpack combo
#     ctrs, feat = combo

#     # Load csv
#     df_cogn_list.append(pd.read_csv(
#         OUTDIR + f"cognition/regressors/" \
#         f"pub_meta_cognition_matched_regressors_{feat}_{ctrs}.csv",
#         index_col=0)
#             )


# df_cogn = functools.reduce(
#         lambda left, right: pd.merge(left, right, on=["eid", "age", "sex", "diab", "college", "htn"],
#                                      how="outer"),
#         df_cogn_list
#         )

# df_cogn.shape
