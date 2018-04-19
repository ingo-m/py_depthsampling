# -*- coding: utf-8 -*-

"""Deconvolution of GE fMRI depth profiles (draining effect correction)."""

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


def depth_deconv_06(varNumCon, aryEmp5, lstFctr):
    """
    **Deconvolution of GE fMRI depth profiles (draining effect correction)**.

    Function of the model-based correction of draining effect pipeline.

    Function of the depth sampling pipeline.

    Parameters
    ----------
    varNumCon : int
        Number of conditions.

    aryEmp5 : np.array
        Two-dimensional array with depth profiles defined at 5 depth levels,
        separately for each condition: aryEmp5[condition, depthlevel].

    lstFctr : list
        List of fractions of underestimation of empirical deep GM signal. For
        instance, a value of 0.1 simulates that the deep GM signal was
        understimated by 10%, and the deepest signal level will be multiplied
        by 1.1. Each factor will be represented by a sepearate line in the
        plot.

    Returns
    -------
    aryNrn : np.array
        Three-dimensional array with depth profiles corrected after applying
        the underestimation factors. Dimensions: aryNrn[idxFctr,
        idxCondition, idxDepth]. The first dimension corresponds to the
        weighting factor (for instance, if three weighting factors will be
        provided, like `lstFctr=[0.1, 0.2, 0.3]`, there will be three values
        along this dimensions).

    Notes
    -----
    The purpose of this function is to test the effect of a systematic
    underestimation of local activity at the deepest depth level (close to WM)
    on the deconvolution (draining effect removal). Same as 'model 1', i.e.
    only correcting draining effect, but the local signal at the deepest depth
    level is multiplied with a factor to simulate a higher signal in deep GM.
    The rational for this is that due to partial volume effects and/or
    segmentation errors, the local signal at the deepest depth level may have
    been underestimated.

    Reference
    ---------
    Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular
    model for examining the specificity of the laminar BOLD signal.
    Neuroimage, 132, 491-498.
    """
    # *** Deconvolution with random noise

    print('------Deconvolution - Model 6 (underestimation of deep GM signal)')

    # *** Deconvolution with deep-GM underestimation factors

    # Number of scaling factors:
    varNumFct = len(lstFctr)

    # Array for deconvolved depth profiles:
    aryNrn = np.zeros((varNumFct, aryEmp5.shape[0], aryEmp5.shape[1]))

    # Loop through deep-GM-signal-intensity-scaling-factors:
    for idxFctr in range(varNumFct):

        # Layer VI:
        aryNrn[idxFctr, :, 0] = np.multiply(aryEmp5[:, 0],
                                            (lstFctr[idxFctr] + 1.0)
                                            )

        # Layer V:
        aryNrn[idxFctr, :, 1] = (aryEmp5[:, 1]
                                 - (0.6 / 1.9) * aryNrn[idxFctr, :, 0])

        # Layer IV:
        aryNrn[idxFctr, :, 2] = (aryEmp5[:, 2]
                                 - (0.3 / 1.5) * aryNrn[idxFctr, :, 1]
                                 - (0.6 / 1.9) * aryNrn[idxFctr, :, 0])

        # Layer II/III:
        aryNrn[idxFctr, :, 3] = (aryEmp5[:, 3]
                                 - (1.3 / 2.2) * aryNrn[idxFctr, :, 2]
                                 - (0.3 / 1.5) * aryNrn[idxFctr, :, 1]
                                 - (0.5 / 1.9) * aryNrn[idxFctr, :, 0])

        # Layer I:
        aryNrn[idxFctr, :, 4] = (aryEmp5[:, 4]
                                 - (0.7 / 1.7) * aryNrn[idxFctr, :, 3]
                                 - (1.3 / 2.2) * aryNrn[idxFctr, :, 2]
                                 - (0.3 / 1.5) * aryNrn[idxFctr, :, 1]
                                 - (0.5 / 1.9) * aryNrn[idxFctr, :, 0])

    return aryNrn
