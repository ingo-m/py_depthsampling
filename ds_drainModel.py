"""
**Model-based correction of draining effect**.

Function of the depth sampling pipeline.

Notes
-----

The purpose of this script is to remove the contribution of lower cortical
depth levels to the signal at each consecutive depth level. In other words,
at a given depth level, the contribution from lower depth levels is removed
based on the model proposed by Markuerkiaga et al. (2016).

The correction for the draining effect is done in a function called by this
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
from scipy.interpolate import griddata
from ds_pltAcrSubsMean import funcPltAcrSubsMean
from ds_drainModelDecon01 import depth_deconv_01
from ds_drainModelDecon02 import depth_deconv_02
from ds_drainModelDecon03 import depth_deconv_03
from ds_drainModelDecon04 import depth_deconv_04
from ds_findPeak import find_peak


# ----------------------------------------------------------------------------
# *** Define parameters

# Which draining model to use (1, 2, 3, or 4 - see above for details):
varMdl = 4

# ROI (V1 or V2):
strRoi = 'v2'

# Path of depth-profile to correct:
strPthPrf = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/{}.npy'  #noqa
strPthPrf = strPthPrf.format(strRoi)

# Output path for corrected depth-profiles:
strPthPrfOt = '/home/john/PhD/ParCon_Depth_Data/Higher_Level_Analysis/{}_corrected_model_{}.npy'  #noqa
strPthPrfOt = strPthPrfOt.format(strRoi, str(varMdl))

# Output path & prefix for plots:
strPthPltOt = '/home/john/PhD/Tex/deconv/{}_model_{}/deconv_{}_m{}_'  #noqa
strPthPltOt = strPthPltOt.format(strRoi, str(varMdl), strRoi, str(varMdl))

# File type suffix for plot:
strFlTp = '.svg'

# Figure scaling factor:
varDpi = 80.0

# Limits of y-axis for across subject plot:
varAcrSubsYmin01 = -0.05
varAcrSubsYmax01 = 2.0
varAcrSubsYmin02 = -0.05
varAcrSubsYmax02 = 2.0
# varAcrSubsYmin01 = -0.05
# varAcrSubsYmax01 = 800.0
# varAcrSubsYmin02 = -0.05
# varAcrSubsYmax02 = 800.0

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Condition labels:
lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']

# Parameters specific to 'model 4' (i.e. random noise model):
if varMdl == 4:
    # How many random noise samples:
    varNumIt = 10000
    # How much noise (SD of Gaussian distribution to sample noise from, percent
    # of noise to multiply the signal with):
    varSd = 0.1


# ----------------------------------------------------------------------------
# *** Load depth profile from disk

print('-Model-based correction of draining effect')

print('---Loading data')

# Array with single-subject depth sampling results, of the form
# aryEmpSnSb[idxSub, idxCondition, idxDpth].
aryEmpSnSb = np.load(strPthPrf)

# Number of subjects:
varNumSub = aryEmpSnSb.shape[0]

# Number of conditions:
varNumCon = aryEmpSnSb.shape[1]

# Number of equi-volume depth levels in the input data:
varNumDpth = aryEmpSnSb.shape[2]


# ----------------------------------------------------------------------------
# *** Loop through subjects

print('---Subject-by-subject deconvolution')

# Array for single-subject interpolation result (before deconvolution):
aryEmp5SnSb = np.zeros((varNumSub, varNumCon, 5))

# Array for single-subject deconvolution result:
if varMdl != 4:
    aryNrnSnSb = np.zeros((varNumSub, varNumCon, 5))

elif varMdl == 4:
    # The array for single-subject deconvolution result has an additional
    # dimension in case of model 4 (number of iterations):
    aryNrnSnSb = np.zeros((varNumSub, varNumIt, varNumCon, 5))
    # Generate random noise for model 4:
    aryNseRnd = np.random.randn(varNumIt, varNumCon, varNumDpth)
    # Scale variance:
    aryNseRnd = np.multiply(aryNseRnd, varSd)
    # Centre at one:
    aryNseRnd = np.add(aryNseRnd, 1.0)

for idxSub in range(0, varNumSub):

    # -------------------------------------------------------------------------
    # *** Interpolation (downsampling)

    # The empirical depth profiles are defined at more depth levels than the
    # draining model. We downsample the empirical depth profiles to the number
    # of depth levels of the model.

    # The relative thickness of the layers differs between V1 & V2.
    if strRoi == 'v1':
        print('------Interpolation - V1')
        # Relative thickness of the layers (layer VI, 20%; layer V, 10%; layer
        # IV, 40%; layer II/III, 20%; layer I, 10%; Markuerkiaga et al. 2016).
        # lstThck = [0.2, 0.1, 0.4, 0.2, 0.1]
        # From the relative thickness, we derive the relative position of the
        # layers (we set the position of each layer to the sum of all lower
        # layers plus half  its own thickness):
        vecPosMdl = np.array([0.1, 0.25, 0.5, 0.8, 0.95])

    elif strRoi == 'v2':
        print('------Interpolation - V2')
        # Relative position of the layers (accordign to Weber et al., 2008,
        # Figure 5C, p. 2322). We start with the absolute depth:
        vecPosMdl = np.array([160.0, 590.0, 1110.0, 1400.0, 1620.0])
        # Divide by overall thickness (1.7 mm):
        vecPosMdl = np.divide(vecPosMdl, 1700.0)

    # Position of empirical datapoints:
    vecPosEmp = np.linspace(0.0, 1.0, num=varNumDpth, endpoint=True)

    # Vector for downsampled empirical depth profiles:
    aryEmp5 = np.zeros((varNumCon, 5))

    # Loop through conditions and downsample the depth profiles:
    for idxCon in range(0, varNumCon):
        # Interpolation:
        aryEmp5[idxCon] = griddata(vecPosEmp,
                                   aryEmpSnSb[idxSub, idxCon, :],
                                   vecPosMdl,
                                   method='linear')

    # Put interpolation result for this subject into the array:
    aryEmp5SnSb[idxSub, :, :] = np.copy(aryEmp5)

    # -------------------------------------------------------------------------
    # *** Subtraction of draining effect

    # (1) Deconvolution based on Markuerkiaga et al. (2016).
    if varMdl == 1:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_01(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :])

    # (2) Deconvolution based on Markuerkiaga et al. (2016) & scaling based on
    #     Markuerkiaga et al. (2016).
    elif varMdl == 2:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_02(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :])

    # (3) Deconvolution based on Markuerkiaga et al. (2016) & scaling based on
    #     Weber et al. (2008).
    elif varMdl == 3:
        aryNrnSnSb[idxSub, :, :] = depth_deconv_03(varNumCon,
                                                   aryEmp5SnSb[idxSub, :, :],
                                                   strRoi=strRoi)

    # (4) Deconvolution based on Markuerkiaga et al. (2016), with random error.
    elif varMdl == 4:
            aryNrnSnSb[idxSub, :, :, :] = depth_deconv_04(varNumCon,
                                                          aryEmp5,
                                                          aryNseRnd)


    # -------------------------------------------------------------------------
    # *** Normalisation

    # Calculate 'grand mean', i.e. the mean PE across depth levels and
    # conditions:
    # varGrndMean = np.mean(aryNrnSnSb[idxSub, :, :])

    # Divide all values by the grand mean:
    # aryNrnSnSb[idxSub, :, :] = np.divide(aryNrnSnSb[idxSub, :, :],
    #                                      varGrndMean)


# ----------------------------------------------------------------------------
# *** Save corrected depth profiles

# Save array with single-subject corrected depth profiles, of the form
# aryNrnSnSb[idxSub, idxCondition, idxDpth].
np.save(strPthPrfOt,
        aryNrnSnSb)


# ----------------------------------------------------------------------------
# *** Peak positions percentile bootstrap

# Bootstrapping in order to obtain an estimate of across-subjects variance is
# not performed for model 4. (Model 4 is used to test the effect or random
# error in the model assumptions.)
if varMdl != 4:

    print('---Peak positions in depth profiles - percentile bootstrap')

    # We bootstrap the peak finding. Peak finding needs to be performed both
    # before and after deconvolution, separately for all stimulus conditions.

    # Number of resampling iterations:
    varNumIt = 10000

    # Lower & upper bound of percentile bootstrap (in percent):
    varCnfLw = 2.5
    varCnfUp = 97.5

    # Random array with subject indicies for bootstrapping of the form
    # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the
    # subjects to the sampled on that iteration.
    aryRnd = np.random.randint(0,
                               high=varNumSub,
                               size=(varNumIt, varNumSub))

    # Loop before/after deconvolution:
    for idxDec in range(2):

        if idxDec == 0:
            print('------UNCORRECTED')
        if idxDec == 1:
            print('------CORRECTED')

        # Array for peak positins before deconvolution, of the form
        # aryPks01[idxCondition, idxIteration]
        aryPks01 = np.zeros((2, varNumCon, varNumIt))

        # Array for actual bootstrap samples:
        aryBoo = np.zeros((varNumIt, 5))

        # Loop through conditions:
        for idxCon in range(0, varNumCon):

            # Create array with bootstrap samples:
            for idxIt in range(0, varNumIt):

                # Before deconvolution:
                if idxDec == 0:
                    # Take mean across subjects in bootstrap samples:
                    aryBoo[idxIt, :] = np.mean(aryEmp5SnSb[aryRnd[idxIt, :],
                                                           idxCon,
                                                           :],
                                               axis=0)

                # After deconvolution:
                if idxDec == 1:
                    # Take mean across subjects in bootstrap samples:
                    aryBoo[idxIt, :] = np.mean(aryNrnSnSb[aryRnd[idxIt, :],
                                                          idxCon,
                                                          :],
                                               axis=0)

            # Find peaks:
            aryPks01[idxDec, idxCon, :] = find_peak(aryBoo,
                                                    varNumIntp=100,
                                                    varSd=0.05,
                                                    lgcStat=False)

            # Median peak position:
            varTmpMed = np.median(aryPks01[idxDec, idxCon, :])

            # Confidence interval (percentile bootstrap):
            varTmpCnfLw = np.percentile(aryPks01[idxDec, idxCon, :], varCnfLw)
            varTmpCnfUp = np.percentile(aryPks01[idxDec, idxCon, :], varCnfUp)

            # Print result:
            strTmp = ('---------Median peak position: '
                      + str(np.around(varTmpMed, decimals=2)))
            print(strTmp)
            strTmp = ('---------Percentile bootstrap '
                      + str(np.around(varCnfLw, decimals=1))
                      + '%: '
                      + str(np.around(varTmpCnfLw, decimals=2)))
            print(strTmp)
            strTmp = ('---------Percentile bootstrap '
                      + str(np.around(varCnfUp, decimals=1))
                      + '%: '
                      + str(np.around(varTmpCnfUp, decimals=2)))
            print(strTmp)


# ----------------------------------------------------------------------------
# *** Plot results

print('---Plot results')

if varMdl != 4:

    # Plot across-subjects mean before deconvolution:
    strTmpTtl = '{} before deconvolution'.format(strRoi.upper())
    strTmpPth = (strPthPltOt + 'before_')
    funcPltAcrSubsMean(aryEmp5SnSb,
                       varNumSub,
                       5,
                       varNumCon,
                       varDpi,
                       varAcrSubsYmin01,
                       varAcrSubsYmax01,
                       lstConLbl,
                       strXlabel,
                       strYlabel,
                       strTmpTtl,
                       strTmpPth,
                       strFlTp,
                       strErr='conf95')

    # Across-subjects mean after deconvolution:
    strTmpTtl = '{} after deconvolution'.format(strRoi.upper())
    strTmpPth = (strPthPltOt + 'after_')
    funcPltAcrSubsMean(aryNrnSnSb,
                       varNumSub,
                       5,
                       varNumCon,
                       varDpi,
                       varAcrSubsYmin02,
                       varAcrSubsYmax02,
                       lstConLbl,
                       strXlabel,
                       strYlabel,
                       strTmpTtl,
                       strTmpPth,
                       strFlTp,
                       strErr='conf95')

elif varMdl == 4:

    # For 'model 4', i.e. the random noise model, we are interested in the
    # variance across random-noise iterations. We are *not* interested in the
    # variance across subjects in this case. Because we used the same random
    # noise across subjects, we can average over subjects.
    aryNrnSnSb = np.mean(aryNrnSnSb, axis=0)

    # Across-subjects mean after deconvolution:
    strTmpTtl = '{} after deconvolution'.format(strRoi.upper())
    strTmpPth = (strPthPltOt + 'after_')
    funcPltAcrSubsMean(aryNrnSnSb,
                       varNumSub,
                       5,
                       varNumCon,
                       varDpi,
                       varAcrSubsYmin02,
                       varAcrSubsYmax02,
                       lstConLbl,
                       strXlabel,
                       strYlabel,
                       strTmpTtl,
                       strTmpPth,
                       strFlTp,
                       strErr='prct95')


# ----------------------------------------------------------------------------

print('-Done.')
