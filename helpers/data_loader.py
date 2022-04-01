#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:58:02 2021

@author: botond
"""

import functools
import pandas as pd
import numpy as np
from scipy import stats, spatial
from IPython import get_ipython

class DataLoader:

    def __init__(self):
        pass

    def load_basic_vars(self, SRCDIR):

        """
        Load regressors probably every analysis will use one way or the other
        """

        # Age
        self.age = pd.read_csv(SRCDIR + "ivs/age.csv", index_col=0) \
            [["eid", "age-2"]] \
            .rename({"age-2": "age"}, axis=1)

        # Sex
        self.sex = pd \
            .read_csv(SRCDIR + "ivs/sex.csv", index_col=0)[["eid", "sex"]] \
            .set_axis(["eid", "sex"], axis=1)

        # Diabetes diagnosis
        self.diab = pd.read_csv(SRCDIR + "ivs/diab.csv", index_col=0) \
            [["eid", "diab-2"]] \
            .rename({"diab-2": "diab"}, axis=1) \
            .query('diab in [0, 1]')

        # College
        self.college = pd.read_csv(SRCDIR + "ivs/college.csv", index_col=0) \
            [["eid", "college"]]

        # BMI
        self.bmi = pd.read_csv(SRCDIR + "ivs/bmi.csv", index_col=0) \
            .rename({"bmi-2": "bmi"}, axis=1) \
            [["eid", "bmi"]] \
            .dropna()

        # Menopause
        self.mp = pd \
           .read_csv(SRCDIR + "ivs/mp.csv", index_col=0) \
           .pipe(lambda df: df.assign(**{
              "mp": df.eval('mp_0 == 1 | mp_1 == 1 | mp_2 == 1').astype(int)})) \

        # Hormone replacement therapy
        self.hrt = pd.read_csv(SRCDIR + "ivs/hrt_age.csv", index_col=0) \
            .rename({"hrt_age_2": "hrt"}, axis=1) \
            [["eid", "hrt"]]

        # Age of diabetes diagnosis (rough estimate!, averaged)
        self.age_onset = pd \
            .read_csv(SRCDIR + "ivs/age_onset.csv", index_col=0) \
            .set_index("eid") \
            .mean(axis=1) \
            .rename("age_onset") \
            .reset_index()

        # Systolic blood pressure
        self.sbp = pd \
            .read_csv(SRCDIR + "ivs/sbpa.csv", index_col=0) \
            .merge(pd.read_csv(SRCDIR + "ivs/sbpm.csv", index_col=0),
                   on="eid", how="outer") \
            .set_index("eid") \
            [["sbpa_2", "sbpa_2.1", "sbpm_2", "sbpm_2.1"]] \
            .mean(axis=1) \
            .rename("sbp") \
            .reset_index() \
            .dropna()

        # Diastolic blood pressure
        self.dbp = pd \
            .read_csv(SRCDIR + "ivs/dbpa.csv", index_col=0) \
            .merge(pd.read_csv(SRCDIR + "ivs/dbpm.csv", index_col=0),
                   on="eid", how="outer") \
            .set_index("eid") \
            [["dbpa_2", "dbpa_2.1", "dbpm_2", "dbpm_2.1"]] \
            .mean(axis=1) \
            .rename("dbp") \
            .reset_index() \
            .dropna()

        # Hypertension based on sbp and dbp
        self.htn = pd \
            .merge(self.sbp, self.dbp, on="eid", how="inner")\
            .set_index("eid") \
            .eval('sbp >= 140 | dbp >= 90') \
            .astype(int) \
            .rename("htn") \
            .reset_index()

        # Diabetes duration
        self.duration = \
            (self.age.set_index("eid")["age"] - \
             self.age_onset.set_index("eid")["age_onset"]) \
            .rename("duration") \
            .reset_index()

        return self


    def filter_vars(self, AGE_CO, T1DM_CO):
        """
        Filter out samples based on criteria across given variables
        """

        # Remove subjects below the age threshold
        self.age = self.age.query(f'age > {AGE_CO}')

        # Remove female subjects who did not report menopause
        self.mp = self.mp \
           .merge(self.sex, on="eid", how="right") \
           .query('(sex==0 & mp==1) or (sex==1 & mp!=mp)') \
           [["eid", "mp"]]

        # Remove female subjects who have reported ongoing hrt at the time
        self.hrt = self.hrt \
            .merge(self.sex, on="eid", how="right") \
            .query('(sex==0 & hrt != -11) or (sex==1 & hrt!=hrt)') \
            .pipe(lambda df: df.assign(**{"hrt": 1})) \
           [["eid", "hrt"]]

        # Remove diabetic subjects with missing age of onset OR have
        # too early age of onset which would suggest T1DM. If age of onset is below
        # T1DM_CO, subject is excluded.
        self.age_onset = self.age_onset \
            .merge(self.diab, on="eid", how="inner") \
            .query(f'(diab==0 & age_onset!=age_onset) or '\
                   f'(diab==1 & age_onset>={T1DM_CO})') \
            [["eid", "age_onset"]]

        return self

