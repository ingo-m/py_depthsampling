# -*- coding: utf-8 -*-
"""Deconvolution of GE fMRI depth profiles (draining effect correction)."""

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


def deconv_07(varNumCon, aryEmp5):
    """
    Deconvolution of GE fMRI depth profiles (draining effect correction).

    Function of the model-based correction of draining effect pipeline.

    Function of the depth sampling pipeline.

    Parameters
    ----------
    varNumCon : int
        Number of conditions.

    aryEmp5 : np.array
        Two-dimensional array with depth profiles defined at 5 depth levels,
        separately for each condition: aryEmp5[condition, depthlevel].

    Returns
    -------
    aryNrn : np.array
        Two-dimensional array with corrected depth profiles (same dimensions
        as the input array: aryNrn[condition, depthlevel]).

    Notes
    -----
    The purpose of this script is to remove the contribution of lower cortical
    depth levels to the signal at each consecutive depth level. In other
    words, at a given depth level, the contribution from lower depth levels is
    removed based on the model proposed by Markuerkiaga et al. (2016).

    **Same as model 02, but using matrix inversion instead of iterative
    subtraction.**

    In other words, if the neuronal signal at each layer was the same, this
    would result in different fMRI signal strength at each layer *even without
    the draining effect*, according to the model proposed by Markuerkiaga et
    al. (2016). This function accounts *only* for the draining effect, and
    does *not* correct for this additional difference in fMRI signal strength
    due to differences in vascular density and/or haemodynamic coupling.

    Let `varEmp*` be the empirically measured fMRI signal at different
    cortical depth levels, and `varCrct*` be the corrected signal. Following
    the model by Markuerkiaga et al. (2016), the fMRI signal for each cortical
    layer for a GE sequence  is corrected for the draining effect as follows:

...

    Reference
    ---------
    Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular
    model for examining the specificity of the laminar BOLD signal.
    Neuroimage, 132, 491-498.
    """
    # *** Deconvolution (removal of draining effect)

    print('------Deconvolution - Model 7')

    # Array with physiological point spread function according to Markuerkiaga
    # et al. (2016):
    aryPsf = np.ones((5, 5))

    # PSF from layer VI:
    aryPsf[0, :] = [1.9, 0.6, 0.6, 0.5, 0.5]

    # PSF from layer V:
    aryPsf[1, :] = [1.0, 1.5, 0.3, 0.3, 0.3]

    # PSF from layer IV:
    aryPsf[2, :] = [1.0, 1.0, 2.2, 1.3, 1.3]

    # PSF from layer II/III:
    aryPsf[3, :] = [1.0, 1.0, 1.0, 1.7, 0.7]

    # PSF from layer I:
    aryPsf[4, :] = [1.0, 1.0, 1.0, 1.0, 1.6]

    # Array for deconvolved depth profiles:
    aryNrn = np.zeros((varNumCon, 5))

    # Loop through conditions and apply deconvolution:
    for idxCon in range(varNumCon):

        aryNrn[idxCon, :] = np.matmul(
                                      np.linalg.inv(aryPsf),
                                      aryEmp5[idxCon, :].reshape((5, 1))
                                      ).flatten()

        # varRatio = np.divide(aryEmp5[idxCon, 0], aryNrn[idxCon, 0])
        # aryNrn[idxCon, :] = np.multiply(aryNrn[idxCon, :], varRatio)
        
        # varDiff = np.subtract(aryEmp5[idxCon, 0], aryNrn[idxCon, 0])
        # aryNrn[idxCon, :] = np.add(aryNrn[idxCon, :], varDiff)

    return aryNrn
