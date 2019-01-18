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
from ds_crfParPerm01 import crf_par_perm_01


def perm_hlf_max_peak(objDpth01,
                      objDpth02,
                      vecEmpX,
                      strFunc='power',
                      varNumIt=1000,
                      varPar=10):
    """
    Parent function for parallelised resampling (permutation) & CRF fitting.

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
    vecEmpX : np.array
        Empirical x-values at which model will be fitted (e.g. stimulus
        contrast levels at which stimuli were presented), of the form
        vecEmpX[idxCon].
    strFunc : str
        Which contrast response function to fit. 'power' for power function, or
        'hyper' for hyperbolic ratio function.
    varNumIt : int or None
        Number of resampling iterations. Set to `None` in case of small enough
        sample size for exact test (i.e. all possible resamples), otherwise
        Monte Carlo resampling is performed.
    varPar : int
        Number of process to run in parallel.

    Returns
    -------
    aryDpth01 : np.array
        Array with single-subject cortical depth profiles for first condition
        (e.g. V1), of the form: aryDpth[idxSub, idxCondition, idxDpth].
    aryDpth02 : np.array
        Array with single-subject cortical depth profiles for second condition
        (e.g. V2), of the form: aryDpth[idxSub, idxCondition, idxDpth].
    aryMdlY : np.array
        Fitted y-values (predicted response based on CRF model), of the form
        aryMdlY[idxRoi, idxIteration, idxDpt, idxContrast]
    aryHlfMax : np.array
        Predicted response at 50 percent contrast based on CRF model. Array of
        the form aryHlfMax[idxRoi, idxIteration, idxDpt].
    arySemi : np.array
        Semisaturation contrast (predicted contrast needed to elicit 50 percent
        of the response amplitude that would be expected with a 100 percent
        contrast stimulus). Array of the form
        arySemi[idxRoi, idxIteration, idxDpt].
    aryRes : np.array
        Residual variance at empirical contrast levels. Array of the form
        aryRes[idxRoi, idxIteration, idxCondition, idxDpt].

    Notes
    -----
    Parent function for parallelised resampling with replacement (i.e.
    permutation) & CRF fitting for two sets of depth profiles. The labels of
    the two sets of depth profiles (e.g. 'V1' and 'V2') are randomly resampled
    within subjects (because within subject profiles are assumed to be
    dependent). CRF fitting is performed on the resampled profiles, for each
    resampling iteration. Further analysis (e.g. peak identification) can be
    performed on the output of this function.

    Function of the depth sampling pipeline.
    """
    # ------------------------------------------------------------------------
    # *** Prepare resampling

    # ** Permutation label 1 (e.g. V1):

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

    # ** Permutation label 2 (e.g. V2):

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
    # *** Return
    return aryDpth01, aryDpth02, aryMdlY, aryHlfMax, arySemi, aryRes
