# -*- coding: utf-8 -*-
"""
Model-based correction of draining effect.

Function of the depth sampling pipeline.

Notes
-----

Remove the contribution of lower cortical depth levels to the signal at each
consecutive depth level. In other words, at a given depth level, the
contribution from lower depth levels is removed based on the model proposed by
Markuerkiaga et al. (2016).

The correction for the draining effect is applyed by a function called by this
script. There are three different option for correction (see respective
functions for details):

(1) Only correct draining effect (based on model by Markuerkiaga et al., 2016).

(2) Correct draining effect (based on model by Markuerkiaga et al., 2016) &
    perform scaling to account for different vascular density and/or
    haemodynamic coupling between depth levels based on model by Markuerkiaga
    et al. (2016).

(3) Correct draining effect (based on model by Markuerkiaga et al., 2016) &
    perform scaling to account for different vascular density and/or
    haemodynamic coupling between depth levels based on data by Weber et al.
    (2008). This option allows for different correction for V1 & extrastriate
    cortex.

(4) Same as (1), i.e. only correcting draining effect, but with Gaussian random
    error added to the draining effect assumed by Markuerkiaga et al. (2016).
    The purpose of this is to test how sensitive the results are to violations
    of the model assumptions. If this solution is selected, the error bars in
    the plots do not represent the bootstrapped across-subjects variance, but
    the variance across iterations of random-noise iterations.

(5) Similar to (4), but two different types of error are simulated: (1) random
    error, sampled from a Gaussian distribution, and (2) systematic error. If
    this solution is selected, only the depth profiles for one condition are
    plotted. The error shading in the plots does then not represent the
    bootstrapped across-subjects variance, but the variance across iterations
    of random-noise iterations. In addition, two separate lines correspond to
    the systematic error (lower and upper bound). The random noise is different
    across depth levels, whereas the systematic noise uses one fixed factor
    across all depth levels, reflecting a hypothetical general bias of the
    model.

(6) Similar to (5), but in this version the effect of a systematic
    underestimation of local activity at the deepest depth level (close to WM)
    is tested. The rational for this is that due to partial volume effects
    and/or segmentation errors, the local signal at the deepest depth level
    may have been underestimated. Thus, here we simulate the profile shape
    after deconvolution with substantially higher signal at the deepest GM
    level.

The following data from Markuerkiaga et al. (2016) is used in this script,
irrespective of which draining effect model is choosen:

    "The cortical layer boundaries of human V1 in the model were fixed
    following de Sousa et al. (2010) and Burkhalter and Bernardo (1989):
    layer VI, 20%;
    layer V, 10%;
    layer IV, 40%;
    layer II/III, 20%;
    layer I, 10%
    (values rounded to the closest multiple of 10)." (p. 492)

References
----------
Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular model
    for examining the specificity of the laminar BOLD signal. Neuroimage, 132,
    491-498.

Weber, B., Keller, A. L., Reichold, J., & Logothetis, N. K. (2008). The
    microvascular system of the striate and extrastriate visual cortex of the
    macaque. Cerebral Cortex, 18(10), 2318-2330.
"""

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


from py_depthsampling.drain_model.drain_model_main import drain_model

# -----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to use (1, 2, 3, 4, 5, or 6 - see above for details):
lstMdl = [1]

# Meta-condition (within or outside of retinotopic stimulus area):
lstMetaCon = ['stimulus', 'periphery']

# ROI ('v1', 'v2' or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Hemisphere ('rh' or 'lh'):
lstHmsph = ['lh', 'rh']

# Path of depth-profile to correct (meta-condition, ROI, hemisphere, and
# condition left open):
strPthPrf = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}.npy'  #noqa

# Output path for corrected depth-profiles (meta-condition, ROI, hemisphere,
# condition, and model index left open):
strPthPrfOt = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/{}/{}_{}_{}_deconv_model_{}.npy'  #noqa

# Output path & prefix for plots (meta-condition, ROI, ROI, hemisphere,
# condition, and model index left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/deconv/{}/{}/{}_{}_{}_deconv_model_{}_'  #noqa

# File type suffix for plot:
strFlTp = '.svg'

# Figure scaling factor:
varDpi = 100.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [a.u.]'

# Condition levels (used to complete file names) - nested list:
# lstNstCon = [['Pd', 'Cd', 'Ps'],
#              ['Pd_min_Ps'],
#              ['Pd_min_Cd']]
# NOTE: Higher level contrast should be calculated again after deconvolution.
lstNstCon = [['Pd_sst', 'Cd_sst', 'Ps_sst'],
             ['Pd_trn', 'Cd_trn', 'Ps_trn']]

# Condition labels:
# lstNstConLbl = [['PacMan Dynamic', 'Control Dynamic', 'PacMan Static'],
#                 ['PacMan D - PacMan S'],
#                 ['PacMan D - Control D']]
lstNstConLbl = [['PacMan Dynamic Sustained',
                 'Control Dynamic Sustained',
                 'PacMan Static Sustained'],
                ['PacMan Dynamic Transient',
                 'Control Dynamic Transient',
                 'PacMan Static Transient']]

# Number of resampling iterations for peak finding (for models 1, 2, and 3) or
# random noise samples (models 4 and 5):
varNumIt = 10000

# Lower & upper bound of percentile bootstrap (in percent), for bootstrap
# confidence interval (models 1, 2, and 3) - this value is only printed, not
# plotted - or plotted confidence intervals in case of model 5:
varCnfLw = 0.5
varCnfUp = 99.5

# Parameters specific to 'model 4' (i.e. random noise model) and 'model 5'
# (random & systematic error model):
# if (varMdl == 4) or (varMdl == 5):
# Extend of random noise (SD of Gaussian distribution to sample noise from,
# percent of noise to multiply the signal with):
varNseRndSd = 0.15
# Extend of systematic noise (only relevant for model 5):
varNseSys = 0.3

# Parameters specific to 'model 6' (simulating underestimation of deep GM
# signal):
# if varMdl == 6:
# List of fraction of underestimation of empirical deep GM signal. For
# instance, a value of 0.1 simulates that the deep GM signal was
# understimated by 10%, and the deepest signal level will be multiplied by
# 1.1. (Each factor will be represented by a sepearate line in the plot.)
lstFctr = [0.0, 0.25, 0.5, 0.75]
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Loop through ROIs / conditions

# Loop through models, ROIs, hemispheres, and conditions to create plots:
for idxMtaCn in range(len(lstMetaCon)):  #noqa
    for idxMdl in range(len(lstMdl)):  #noqa
        for idxRoi in range(len(lstRoi)):
            for idxHmsph in range(len(lstHmsph)):
                for idxCon in range(len(lstNstCon)):

                    # Limits of axes need to be adjusted based on ROI,
                    # condition, hemisphere.

                    # Limits of y-axis for ACROSS SUBJECT PLOTS:

                    if idxRoi == 0:  # v1
                        if idxCon == 0:  # v1 simple contrasts
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                # Limits of y-axis for across subject plot:
                                varAcrSubsYmin01 = -400.0
                                varAcrSubsYmax01 = 0.0
                                varAcrSubsYmin02 = -400.0
                                varAcrSubsYmax02 = 0.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                # Limits of y-axis for across subject plot:
                                varAcrSubsYmin01 = 0.0
                                varAcrSubsYmax01 = 600.0
                                varAcrSubsYmin02 = 0.0
                                varAcrSubsYmax02 = 600.0

                    elif idxRoi == 1:  # v2
                        if idxCon == 0:  # v2 simple contrasts
                            if lstMetaCon[idxMtaCn] == 'stimulus':
                                # Limits of y-axis for across subject plot:
                                varAcrSubsYmin01 = -400.0
                                varAcrSubsYmax01 = 0.0
                                varAcrSubsYmin02 = -400.0
                                varAcrSubsYmax02 = 0.0
                            if lstMetaCon[idxMtaCn] == 'periphery':
                                # Limits of y-axis for across subject plot:
                                varAcrSubsYmin01 = 0.0
                                varAcrSubsYmax01 = 600.0
                                varAcrSubsYmin02 = 0.0
                                varAcrSubsYmax02 = 600.0

                    # Call drain model function:
                    drain_model(lstMdl[idxMdl], lstRoi[idxRoi],
                                lstHmsph[idxHmsph],
                                strPthPrf.format(lstMetaCon[idxMtaCn],
                                lstRoi[idxRoi], lstHmsph[idxHmsph], '{}'),
                                strPthPrfOt.format(lstMetaCon[idxMtaCn],
                                lstRoi[idxRoi], lstHmsph[idxHmsph], {},
                                lstMdl[idxMdl]),
                                strPthPltOt.format(lstMetaCon[idxMtaCn],
                                lstRoi[idxRoi], lstRoi[idxRoi],
                                lstHmsph[idxHmsph], lstNstCon[idxCon][0],
                                str(lstMdl[idxMdl])), strFlTp, varDpi,
                                strXlabel, strYlabel, lstNstCon[idxCon],
                                lstNstConLbl[idxCon], varNumIt, varCnfLw,
                                varCnfUp, varNseRndSd, varNseSys, lstFctr,
                                varAcrSubsYmin01, varAcrSubsYmax01,
                                varAcrSubsYmin02, varAcrSubsYmax02)
# -----------------------------------------------------------------------------
