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


def deconv_02(varNumCon, aryEmp5):
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

    **This function removes the draining effect AND divides the local fMRI
    signal at each layer by a (layer specific) constant to account for
    different vascular density and/or neuronal-to-fMRI-signal coupling, based
    on the model by Markuerkiaga et al. (2016).**

    In other words, if the neuronal signal at each layer is the same, this
    would result in different fMRI signal strength at each layer even without
    the draining effect, according to the model proposed by Markuerkiaga et
    al. (2016). This version of the script account both for this effect and
    the draining effect.

    Let varEmpVI, varEmpV, varEmpIV, varEmpII_III, and varEmpI be the observed
    (empirical) signal at the different depth levels, and varNrnVI, varNrnV,
    varNrnIV, varNrnII_III, and varNrnI the underlying neuronal activity.
    Following to the model by Markuerkiaga et al. (2016), the absolute fMRI
    signal for each layer for a GE sequence can be predicted as follows
    (forward model, as depicted in Figure 3F, p. 495):

        Layer VI:

        >>> varEmpVI = 1.9 * varNrnVI

        Layer V:

        >>> varEmpV = 1.5 * varNrnV
                      + 0.6 * varNrnVI

        Layer IV:

        >>> varEmpIV = 2.2 * varNrnIV
                       + 0.3 * varNrnV
                       + 0.6 * varNrnVI

        Layer II/III:

        >>> varEmpII_III = 1.7 * varNrnII_III
                           + 1.3 * varNrnIV
                           + 0.3 * varNrnV
                           + 0.5 * varNrnVI

        Layer I:

        >>> varEmpI = 1.6 * varNrnI
                      + 0.7 * varNrnII_III
                      + 1.3 * varNrnIV
                      + 0.3 * varNrnV
                      + 0.5 * varNrnVI

    These values are translated into the a transfer function to estimate the
    local neural activity at each layer given an empirically observed fMRI
    signal depth profile.

    Reference
    ---------
    Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular
    model for examining the specificity of the laminar BOLD signal.
    Neuroimage, 132, 491-498.
    """
    # *** Deconvolution (removal of draining effect)

    print('------Deconvolution - Model 2')

    # Array for corrected depth profiles:
    aryNrn = np.zeros(aryEmp5.shape)

    # Subtraction of draining effect:
    for idxCon in range(0, varNumCon):

        # Layer VI:
        aryNrn[idxCon, 0] = aryEmp5[idxCon, 0]

        # Layer V:
        aryNrn[idxCon, 1] = (aryEmp5[idxCon, 1]
                             - (0.6 / 1.9) * aryNrn[idxCon, 0])

        # Layer IV:
        aryNrn[idxCon, 2] = (aryEmp5[idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.6 / 1.9) * aryNrn[idxCon, 0])

        # Layer II/III:
        aryNrn[idxCon, 3] = (aryEmp5[idxCon, 3]
                             - (1.3 / 2.2) * aryNrn[idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.5 / 1.9) * aryNrn[idxCon, 0])

        # Layer I:
        aryNrn[idxCon, 4] = (aryEmp5[idxCon, 4]
                             - (0.7 / 1.7) * aryNrn[idxCon, 3]
                             - (1.3 / 2.2) * aryNrn[idxCon, 2]
                             - (0.3 / 1.5) * aryNrn[idxCon, 1]
                             - (0.5 / 1.9) * aryNrn[idxCon, 0])

    # Vector for layer-specific intensity correction (CBV fractions):
    vecCbv = np.array([1.9, 1.5, 2.2, 1.7, 1.6])
    # Normalise the vector to its maximum:
    # vecCbv = np.divide(vecCbv, np.max(vecCbv))

    # Division (correction for different vascular density and/or haemodynamic
    # coupling):
    aryNrn = np.divide(aryNrn, vecCbv[None, :])

    return aryNrn
