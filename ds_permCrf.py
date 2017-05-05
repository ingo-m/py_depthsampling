# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

# Part of py_depthsampling library
# Copyright (C) 2017  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import json

objDpth01 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v1.npy'
objDpth02 = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/v2.npy'
varNumIt=100
varPar=10

def perm_hlf_max_peak(objDpth01,
                      objDpth02,
                      varNumIt=1000,
                      varPar=10):
    """
    Permutation test for difference in peak position for half-maximum response.

    Parameters
    ----------
    objDpth01 : np.array or str
        Array with single-subject cortical depth profiles for first condition
        (e.g. V1), of the form: aryDpth[idxSub, idxCondition, idxDpth]. Either
        a numpy array or a string with the path to an npy file containing the
        array. objDpth01 and objDpth02 must have same dimensions.
    objDpth02 : np.array or str
        Array with single-subject cortical depth profiles for second condition
        (e.g. V2), of the form: aryDpth[idxSub, idxCondition, idxDpth]. Either
        a numpy array or a string with the path to an npy file containing the
        array. objDpth01 and objDpth02 must have same dimensions.

    varNumIt : int
        Number of resampling iterations.
    varPar : int
        Number of process to run in parallel.

    Notes
    -----
    The purpose of this script is to performe a permutation hypothesis test for
    a difference in the peak position in the cortical depth profiles of the
    response at half-maximum contrast between V1 and V2. More specifically, the
    equality of distributions of the peak positions is tested (i.e. the a
    possible difference could be due to a difference in means, variance, or the
    shape of the distribution).

    The procedure is as follow:
    - Condition labels (i.e. V1 & V2) are permuted within subjects for each
      permutation data set (i.e. on each iteration).
    - For each permutation dataset, the mean depth profile of the two groups
      (i.e. randomly created 'V1' and 'V2' assignments) are calculated.
    - The contrast response function (CRF) is fitted for the two mean depth
      profiles.
    - The response at half-maximum contrast is calculated from the fitted CRF.
    - The peak of the half-maximum contrast profile is identified for both
      randomised groups.
    - The mean difference in peak position between the two randomised groups is
      the null distribution.
    - The peak difference on the full profile is calculated, and the
      permutation p-value with respect to the null distribution is produced.

    Function of the depth sampling pipeline.
    """
    # ------------------------------------------------------------------------
    # *** Prepare resampling

    # Test whether the input is a numpy array or a string (with the path to a
    # numpy array):
    lgcAry = (type(objDpth01) == np.ndarray)
    lgcStr = (type(objDpth01) == str)
    # If input is a string, load array from npy file:
    if lgcAry:
        aryDpth01 = objDpth01
    elif lgcStr:
        aryDpth01 = np.load(objDpth01)
    else:
        print(('---Error in bootPlot: input needs to be numpy array or path '
               + 'to numpy array.'))

    # Test whether the input is a numpy array or a string (with the path to a
    # numpy array):
    lgcAry = (type(objDpth02) == np.ndarray)
    lgcStr = (type(objDpth02) == str)
    # If input is a string, load array from npy file:
    if lgcAry:
        aryDpth02 = objDpth02
    elif lgcStr:
        aryDpth02 = np.load(objDpth02)
    else:
        print(('---Error in bootPlot: input needs to be numpy array or path '
               + 'to numpy array.'))


    # ------------------------------------------------------------------------
    # *** Parallelised resampling & CRF fitting

    aryMdlY, aryHlfMax, arySemi, aryRes = crf_par_perm_01(aryDpth01,
                                                          aryDpth02,
                                                          vecEmpX,
                                                          strFunc=strFunc,
                                                          varNumIt=varNumIt,
                                                          varPar=varPar,
                                                          varNumX=1000)

    # ------------------------------------------------------------------------
    # *** Save results

    print('---Saving bootstrapping results as json object')

    # Put results into list:
    lstJson = [aryDpth01,  # Original depth profiles V1
               aryDpth02,  # Original depth profiles V2
               aryMdlY,    # Fitted y-values
               aryHlfMax,  # Predicted response at 50 percent contrast
               arySemi,    # Semisaturation contrast
               aryRes]     # Residual variance

    # Save results to disk:
    with open('data.json', 'w') as objJson:
         json.dump(lstJson, objJson)

    pickle.dump(lstPkl, open(strPthPkl, "wb"))



# Reading data back
with open('data.json', 'r') as f:
     data = json.load(f)


print('-Done.')