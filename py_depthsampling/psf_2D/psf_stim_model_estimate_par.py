# -*- coding: utf-8 -*-
"""Calculate similarity between visual field projections and stimulus model."""

# Part of py_depthsampling library
# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from scipy.optimize import minimize
from py_depthsampling.psf_2D.utilities_stim_model import psf_diff_stim_mdl


def est_par(idxPrc, aryBooTrgtChnk, vecInit, aryPacMan, aryEdge, aryPeri,
            vecVslX, lstBnds, varScl, queOut):
    """
    Calculate similarity between visual field projections and stimulus model.

    This function is called from `psf_stim_model_estimate` in parallel.

    See `psf_from_stim_model_main.py` for details.
    """
    # Number of bootstrap iterations in this chunk. Shape:
    # `aryBooTrgt[varNumItChnk, varSzeVsm, varSzeVsm], where varNumItChnk are
    # the number of bootstrap samples per chunk.
    varNumItChnk = aryBooTrgtChnk.shape[0]

    # Vectors for bootstrapping distributions (for PSF width, and scaling
    # factors for stimulu centre, edge, and periphery):
    vecBooSd = np.zeros(varNumItChnk)
    vecBooFctCntr = np.zeros(varNumItChnk)
    vecBooFctEdge = np.zeros(varNumItChnk)
    vecBooFctPeri = np.zeros(varNumItChnk)

    # The actual bootstrap PSF model fitting:
    for idxIt in range(varNumItChnk):

        # Fit point spread function:
        dicOptm = minimize(psf_diff_stim_mdl,
                           vecInit,
                           args=(aryPacMan, aryEdge, aryPeri,
                                 aryBooTrgtChnk[idxIt, :, :], vecVslX),
                           bounds=lstBnds)

        # Convert width from array indices to degrees of visual angle:
        varTmpSd = (dicOptm.x[0] / varScl)

        # Bootstrapping results to vector:
        vecBooSd[idxIt] = varTmpSd
        vecBooFctCntr[idxIt] = dicOptm.x[1]
        vecBooFctEdge[idxIt] = dicOptm.x[2]
        vecBooFctPeri[idxIt] = dicOptm.x[3]

    # List for results:
    lstOut = [idxPrc, vecBooSd, vecBooFctCntr, vecBooFctEdge, vecBooFctPeri]

    # Put output to queue:
    queOut.put(lstOut)
