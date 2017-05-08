#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:31:37 2017

@author: john
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

import statsmodels as sm

aryEmpSemi



#endog : 1d array-like
#
#    The dependent variable
#
vecDep = aryEmpSemi.flatten()

#exog : 2d array-like
#
#    A matrix of covariates used to determine the mean structure (the “fixed effects” covariates).
#

vecRoi = np.array((np.zeros(varNumDpt),
                     np.ones(varNumDpt))).flatten()

vecDpth = np.array((np.arange(0, varNumDpt),
                     np.arange(0, varNumDpt))).flatten()

aryInd = np.array((vecRoi,
                   vecDpth))

#groups : 1d array-like
#
#    A vector of labels determining the groups – data from different groups are independent



#A basic mixed model with fixed effects for the columns of exog and a random
# intercept for each distinct value of group:
#
#>>> model = sm.MixedLM(endog, exog, groups)
#>>> result = model.fit()

model = sm.MixedLM(vecDep, vecDpth, vecRoi)



#data = sm.datasets.get_rdataset("dietox", "geepack").data
#md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
#mdf = md.fit()
#print(mdf.summary())
