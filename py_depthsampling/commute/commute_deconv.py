# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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
from scipy.interpolate import griddata
from py_depthsampling.drain_model.drain_model_decon_01 import deconv_01


def commute(aryEmpSnSb, strRoi, varMdl=1):
    """
    Apply spatial deconvolution (drain model).

    Parameters
    ----------
    aryEmpSnSb : np.array
        Array with single subject depth profiles, of the form
        aryEmpSnSb[subject, condition, depth].
    strRoi : string
        Region of interest ('v1' or 'v2').
    varMdl : int
        Which deconvolution model to use. Currently, only 'model 1', based on
        Markuerkiaga et al. (2016) is implemented. If needed, this could easily
        be extended to other models.

    Returns
    -------
    aryDecon : np.array
        Array with deconvolution results, of the form
        aryDecon[subject, condition, depth].

    Apply spatial deconvolution (drain model) as part of a test for commutative
    property of drain model (deconvolution).
    """
    # -------------------------------------------------------------------------
    # *** Preparations
    # Number of subjects:
    varNumSub = aryEmpSnSb.shape[0]

    # Number of conditions:
    varNumCon = aryEmpSnSb.shape[1]

    # Number of equi-volume depth levels in the input data:
    varNumDpth = aryEmpSnSb.shape[2]

    # -------------------------------------------------------------------------
    # *** Subject-by-subject deconvolution

    print('---Subject-by-subject deconvolution')

    # Array for single-subject interpolation result (before deconvolution):
    aryEmp5SnSb = np.zeros((varNumSub, varNumCon, 5))

    # Array for single-subject deconvolution result (defined at 5 depth
    # levels):
    aryDecon5 = np.zeros((varNumSub, varNumCon, 5))
    # Array for deconvolution results in equi-volume space:
    aryDecon = np.zeros((varNumSub, varNumCon, varNumDpth))

    for idxSub in range(0, varNumSub):

        # ---------------------------------------------------------------------
        # *** Interpolation (downsampling)

        # The empirical depth profiles are defined at more depth levels than
        # the draining model. We downsample the empirical depth profiles to the
        # number of depth levels of the model.

        # The relative thickness of the layers differs between V1 & V2.
        if strRoi == 'v1':
            print('------Interpolation - V1')
            # Relative thickness of the layers (layer VI, 20%; layer V, 10%;
            # layer IV, 40%; layer II/III, 20%; layer I, 10%; Markuerkiaga et
            # al. 2016). lstThck = [0.2, 0.1, 0.4, 0.2, 0.1]
            # From the relative thickness, we derive the relative position of
            # the layers (we set the position of each layer to the sum of all
            # lower layers plus half  its own thickness):
            vecPosMdl = np.array([0.1, 0.25, 0.5, 0.8, 0.95])

        elif strRoi == 'v2':
            print('------Interpolation - V2')
            # Relative position of the layers (accordign to Weber et al., 2008,
            # Figure 5C, p. 2322). We start with the absolute depth:
            vecPosMdl = np.array([160.0, 590.0, 1110.0, 1400.0, 1620.0])
            # Divide by overall thickness (1.7 mm):
            vecPosMdl = np.divide(vecPosMdl, 1700.0)

        # Position of empirical datapoints:
        vecPosEmp = np.linspace(np.min(vecPosMdl),
                                np.max(vecPosMdl),
                                num=varNumDpth,
                                endpoint=True)

        # Vector for downsampled empirical depth profiles:
        aryEmp5 = np.zeros((varNumCon, 5))

        # Loop through conditions and downsample the depth profiles:
        for idxCon in range(0, varNumCon):
            # Interpolation:
            aryEmp5[idxCon] = griddata(vecPosEmp,
                                       aryEmpSnSb[idxSub, idxCon, :],
                                       vecPosMdl,
                                       method='cubic')

        # Put interpolation result for this subject into the array:
        aryEmp5SnSb[idxSub, :, :] = np.copy(aryEmp5)

        # ---------------------------------------------------------------------
        # *** Subtraction of draining effect

        # (1) Deconvolution based on Markuerkiaga et al. (2016).
        if varMdl == 1:
            aryDecon5[idxSub, :, :] = deconv_01(varNumCon,
                                                aryEmp5SnSb[idxSub, :, :])

        # ---------------------------------------------------------------------
        # *** Interpolation

        # The original depth profiles were in 'equi-volume' space, and needed
        # to be downsampled in order to apply the deconvolution (because the
        # deconvolution model is defined at a lower number of depth levels than
        # the equivolume space). Here, the results of the deconvolution are
        # brought back into equivolume space. This is advantageous for the
        # creation of depth plots (equal spacing of data points on x-axis), and
        # for the calculation of peak positions (no additional information
        # about relative position of datapoints needs to be passed on).

        # Sampling points for equi-volume space:
        vecIntpEqui = np.linspace(np.min(vecPosMdl),
                                  np.max(vecPosMdl),
                                  num=varNumDpth,
                                  endpoint=True)

        # Loop through conditions:
        for idxCon in range(0, varNumCon):

            # Interpolation back into equi-volume space:
            aryDecon[idxSub, idxCon, :] = griddata(vecPosMdl,
                                                   aryDecon5[idxSub,
                                                             idxCon,
                                                             :],
                                                   vecIntpEqui,
                                                   method='cubic')

    return aryDecon
    # -------------------------------------------------------------------------
