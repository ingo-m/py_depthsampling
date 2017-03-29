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


def depth_deconv_01(varNumCon, aryEmp5):
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

    **This function removes only the draining effect. It does not account for
    different neuronal-to-fMRI-signal coupling in different layers.**

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

    Layer VI:

    >>> varCrctVI = varEmpVI

    Layer V:

    >>> varCrctV = varEmpV
                   - (0.6 / 1.9) * varCrctVI

    Layer IV:

    >>> varCrctIV = varEmpIV
                    - (0.3 / 1.5) * varCrctV
                    - (0.6 / 1.9) * varCrctVI

    Layer II/III:

    >>> varCrctII_III = varEmpII_III
                        - (1.3 / 2.2) * varCrctIV
                        - (0.3 / 1.5) * varCrctV
                        - (0.5 / 1.9) * varCrctVI)

    Layer I:

    >>> varCrctI = varEmpI
                   - (0.7 / 1.7) * varCrctII_III
                   - (1.3 / 2.2) * varCrctIV
                   - (0.3 / 1.5) * varCrctV
                   - (0.5 / 1.9) * varCrctVI

    Reference
    ---------
    Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular
    model for examining the specificity of the laminar BOLD signal.
    Neuroimage, 132, 491-498.
    """
    # ----------------------------------------------------------------------------
    # *** Subtraction of draining effect

    # Array for corrected depth profiles:
    aryNrn = np.zeros(aryEmp5.shape)

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

    return aryNrn
