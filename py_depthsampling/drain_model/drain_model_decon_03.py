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


def depth_deconv_03(varNumCon, aryEmp5, strRoi='v1'):
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
    strRoi : str
        Region of interest. If ``strRoi='v1'``, vascular density and/or
        haemodynamic coupling bias as estimated for V1 is corrected. If
        ``strRoi='v2'``, biased corrected as estimated for extrastriate cortex
        is applied (both based on Weber et al., 2008). Default value is
        ``'v1'``.

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

    **This function divides the local fMRI signal at each layer by a (layer
    specific) constant to account for different vascular density and/or
    neuronal-to-fMRI-signal coupling (based on Weber et al., 2008) AND removes
    the draining effect (based on the model by Markuerkiaga et al., 2016).**

    In other words, if the neuronal signal at each layer is the same, this
    would result in different fMRI signal strength at each layer even without
    the draining effect. This version of the script account both for this
    effect and the draining effect.

    The correction is based on the cortical blood volume as published in
    Weber et al. (2008), Figure 4C (V1) and Figure 5C (extrastriate cortex).
    Cortical blood volume fraction across cortical layers:

    V1:
    Layer I:    2.05
    Layer II:   1.9
    Layer III:  2.0
    Layer IVa:  2.15
    Layer IVb:  2.2
    Layer IVca: 2.6
    Layer IVcb: 2.7
    Layer V:    2.1
    Layer VI:   2.3

    Extrastriate:
    Layer I:      2.0
    Layer II/III: 2.1
    Layer IV:     2.2
    Layer V:      2.1
    Layer VI:     2.0

    The draining model by Markuerkiaga et al. (2016) is only defined at five
    layers (I, II/III, IV, V, VI). Thus, we need to average the above CBV
    fractions within these five layers. We refer to the relative thickness of
    the layers as mentioned in Xing et al. (2012, p. 13875) (refering to
    Hawken et al., 1988):

        "The mean relative thickness of each layer is
        0.442 for layers 2 and 3,
        0.058 for layer 4A,
        0.108 for layer 4B,
        0.083 for layer 4Calpha,
        0.083 for layer 4Cbeta,
        0.11 for layer 5, and
        0.116 for layer 6".

    This leads to the following CBV fractions for the five layers at which the
    draining model is defined:

    V1:

        >>> # Layer VI:
        >>> vecCbv[0] = 2.3
        >>> # Layer V:
        >>> vecCbv[1] = 2.1
        >>>  # Layer IV:
        >>> vecCbv[2] = ((2.15 * 0.058 / 0.332)
        >>>              + (2.2 * 0.108 / 0.332)
        >>>              + (2.6 * 0.083 / 0.332)
        >>>              + (2.7 * 0.083 / 0.332))
        >>> # Layer II/III:
        >>> # (Just the arithmetic mean, because no relative thickness
        >>> # information is given.)
        >>> vecCbv[3] = (1.9 + 2.0) * 0.5
        >>> # Layer I:
        >>> vecCbv[4] = 2.05

    Extrastriate cortex:

        >>> # Layer VI:
        >>> vecCbv[0] = 2.0
        >>> # Layer V:
        >>> vecCbv[1] = 2.1
        >>> # Layer IV:
        >>> vecCbv[2] = 2.2
        >>> # Layer II/III:
        >>> vecCbv[3] = 2.1
        >>> #Layer I:
        >>> vecCbv[4] = 2.0

    These values are translated into the a transfer function to estimate the
    local neural activity at each layer given an empirically observed fMRI
    signal depth profile.

    References
    ----------
    Hawken, M. J., Parker, A. J., & Lund, J. S. (1988). Laminar organization
    and contrast sensitivity of direction-selective cells in the striate
    cortex of the Old World monkey. J. Neurosci., 8(10), 3541-3548.

    Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular
    model for examining the specificity of the laminar BOLD signal.
    Neuroimage, 132, 491-498.

    Weber, B., Keller, A. L., Reichold, J., & Logothetis, N. K. (2008). The
    microvascular system of the striate and extrastriate visual cortex of the
    macaque. Cerebral Cortex, 18(10), 2318-2330.

    Xing, D., Yeh, C. I., Burns, S., & Shapley, R. M. (2012). Laminar analysis
    of visually evoked activity in the primary visual cortex. Proceedings of
    the National Academy of Sciences, 109(34), 13871-13876.
    """
    # ------------------------------------------------------------------------
    # *** Define CBV fractions

    # Vector for layer-specific CBV fractions:
    vecCbv = np.zeros(5)

    # The layer-specific CBV fractions are different for V1 & extrastriate
    # cortex:
    if strRoi == 'v1':

        print('------Calculating layer specific CBV fractions for V1')

        # Layer VI:
        vecCbv[0] = 2.3
        # Layer V:
        vecCbv[1] = 2.1
        # Layer IV:
        vecCbv[2] = ((2.15 * 0.058 / 0.332)
                     + (2.2 * 0.108 / 0.332)
                     + (2.6 * 0.083 / 0.332)
                     + (2.7 * 0.083 / 0.332))
        # Layer II/III:
        vecCbv[3] = (1.9 + 2.0) * 0.5
        # Layer I:
        vecCbv[4] = 2.05

    elif strRoi == 'v2':

        print('------Calculating layer specific CBV fractions for V2')

        # Layer VI:
        vecCbv[0] = 2.0
        # Layer V:
        vecCbv[1] = 2.1
        # Layer IV:
        vecCbv[2] = 2.2
        # Layer II/III:
        vecCbv[3] = 2.1
        # Layer I:
        vecCbv[4] = 2.0

    # Normalise the CBV fraction vector by its maximum value:
    vecCbv = np.divide(vecCbv, np.max(vecCbv))

    # Uncomment the following code and plot aryComparison to compare the
    # correction based on Weber et al. (2008) vs. that of Markuerkiaga et al.
    # (2016).
    #    vecCbvM = np.array([1.9, 1.5, 2.2, 1.7, 1.6])
    #    vecCbvM = np.divide(vecCbvM, np.max(vecCbvM))
    #    aryComparison = np.vstack((vecCbvM, vecCbv)).T

    # ------------------------------------------------------------------------
    # *** Deconvolution (removal of draining effect)

    print('------Deconvolution - Model 3')

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

    # Division (correction for different vascular density and/or haemodynamic
    # coupling):
    aryNrn = np.divide(aryNrn, vecCbv[None, :])

    return aryNrn
