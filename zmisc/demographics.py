#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 17:38:41 2020

@author: botond
"""

import os
import numpy as np
import pandas as pd

# Filepaths
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../")) + "/"
SRCDIR = HOMEDIR + "data/ivs/"
OUTDIR = HOMEDIR + "results/"

class PrepDf:
    """ Class used for interacting with metformin data """

    def __init__(self):
        pass

    def import_data(self, feats):
        """ Import data """

        # Import data
        all_import = [pd.read_csv(SRCDIR + feat + ".csv", index_col=0) \
                      .set_index("eid") \
                      for feat in feats]

        # Unite data for analysis
        self.data = pd.concat(all_import, axis=1)

        # Return
        return self


    def select_cols(self, visit_spec=[None], single=[None], multi=[None],
                    visit=None):
        """ Select a subset of columns """

        # New visit specific columns
        cols_vc = [item + "-" + str(visit) for item in visit_spec]

        # New single visit columns
        cols_s = single

        # New multi columns
        cols_m = [col for multi_col in multi for col in self.data.columns \
                  if multi_col in col]

        self.df = self.data[cols_vc + cols_s + cols_m]

        # Return
        return self

    def rename_cols(self):
        """ Assign name to cols """

        cols = self.df.columns

        # Refine column names
        col_names_pre = ["-".join(col.split("-")[:-1]) if col[-1].isnumeric() \
                    else col for col in cols]

        # If column had multiple occurence, leave the originals
        col_names = col_names_pre.copy()

        # Iterate over all new column names
        for i, name in enumerate(col_names_pre):

            # If occurence is more than 1
            if col_names_pre.count(name) > 1:

                # Get back the original
                col_names[i] = cols[i]

        # Assign new columns
        self.df.columns = col_names

        # Return
        return self

    def clean_ses(self):
        """ Clean socioeconomic status data """

        # Get columns with ses in them
        ses_cols = [col for col in self.df.columns \
                    if col.split("-")[0] == "ses"]

        # Clean out -1 and -3 responses
        for ses_col in ses_cols:

            self.df = self.df.query(f'`{ses_col}` not in [-1, -3]')

        # Return
        return self

    def dropna_df(self, ignore=[None]):
        """ Drop NA entries """

        # Subset of columns to consider
        subset = [col for col in self.df.columns if col not in ignore]

        # Drop Na
        self.df = self.df.dropna(how="any", subset=subset)

        # Return
        return self

    def bin_col(self, feat, bins, precision=3):
        """ Bin a continuous feature """

        # Perform binning
        self.df[feat + "_group"] = pd.cut(self.df[feat], bins,
               include_lowest=True, precision=precision).astype(str)

        # Return
        return self

    def cat_diab(self):
        """ Categorize diabetes """

        # Perform categorizing
        self.df["metf_condition"] = self.df.apply(
                lambda item: "control" if item["diab"] == 0 \
                  else "diab_metf_neg" if (item["diab"] == 1 \
                                                    and item["met"] == 0) \
                  else "diab_metf_pos" if (item["diab"] == 1 \
                                                   and item["met"] == 1) \
                  else None, axis=1)

        # Drop healthy metformin consumers
        self.df = self.df.query('met==0 or (met==1 and diab==1)')

        # Return
        return self

    def query_df(self, qstr):
        """ Perform query on df """

        # Perform query
        self.df = self.df.query(qstr)

        # Return
        return self

    def calc_age_onset(self, clean=True):
        """ Calculate age of onset based on all visits """

        # Compute age of onset
        self.df["age_onset"] = self.df \
                            [["age_onset-0", "age_onset-1", "age_onset-2"]] \
                            .mean(axis=1)

        # Drop raw columns
        self.df = self.df.drop(labels=
                               ["age_onset-0", "age_onset-1", "age_onset-2"],
                               axis=1)

        # Drop diabetic subjects without age_onset or healthy
        # subs with age of onset
        if clean:
            self.df = self.df.query(
                            '(diab==0 & age_onset!=age_onset)' \
                            'or (diab==1 & age_onset==age_onset)')

        # Return
        return self

    def drop_T1DM(self, thr=30):
        """ Drop subjects suspected of having T1DM instead of T2DM """

        # Drop T1DM subs
        self.df = self.df.query(f'~(age_onset < {thr})')

        # Return
        return self

    def comp_duration(self):
        """ Compute duration of T2DM / Time since diagnosis """

        # Compute duration
        self.df["duration"] = self.df["age"] - self.df["age_onset"]

        # Return
        return self

# %%

# T1DM cutoff age
THR = 20

# Transform
dataobj = PrepDf() \
    .import_data(feats=["age", "diab", "sex", "college",
                        "age_onset"]) \
    .select_cols(visit_spec=["age", "diab"],
                 single=["college", "sex"], multi=["age_onset"], visit=2) \
    .rename_cols() \
    .calc_age_onset() \
    .comp_duration() \
    .drop_T1DM(thr=THR) \
    .dropna_df(ignore=["age_onset", "duration"]) \


# Extract data
df = dataobj.df

# Load medication specific data
CTRS = "metfonly_unmed"  # Contrast: diab or age
med = pd.read_csv(SRCDIR + f"../med/{CTRS}.csv")[["eid", CTRS]]
df_med = df.merge(med, on="eid")

# Values
# ------

# Total N
df.shape[0]

## Female
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
