# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

Notes
-----

The purpose of this script is to apply the CRF fitting function to the data
from Tootell et al. (1988), p. 1602, Figure 6. In brief, tracer uptake in
response to prolonged exposure to visual stimuli at four different different
luminance contrast levels (0.08, 0.18, 0.4, 1.0) was measured in layers 3, 4A,
4B, 4Ca, 4Cb, 5, and 6 of Macaque V1. Normalised uptake was (scaled by a factor
of 1000 with respect to original data):

Layer 3:   [70.0, 210.0, 560.0, 570.0]
Layer 4A:  [47.0, 285.0, 550.0, 680.0]
Layer 4B:  [288.0, 341.0, 633.0, 582.0]
Layer 4Ca: [435.0, 461.0, 622.0, 768.0]
Layer 4Cb: [8.0, 117.0, 492.0, 819.0]
Layer 5:   [81.0, 239.0, 432.0, 475.0]
Layer 6:   [455.0, 636.0, 859.0, 996.0]


The profile of semisaturation contrast is plotted with respect to the relative
thickness of layers derived from Hawken et al. 1988, p. 3544, Figure 4. The
relative position of centre of each layer, from 0 at WM/GM interface to 1000
at GM/CSF interface, are:

Layer 6:   55.0
Layer 5:   170.0
Layer 4Cb: 285.0
Layer 4Ca: 362.0
Layer 4B:  450.0
Layer 4A:  530.0
Layer 3:   670.0
Layer 2:   890.0


References
----------
Hawken, M. J., Parker, A. J., & Lund, J. S. (1988). Laminar organization and
    contrast sensitivity of direction-selective cells in the striate cortex of
    the Old World monkey. The Journal of Neuroscience, 8(10), 3541–3548.
Tootell, R. B., Hamilton, S. L., & Switkes, E. (1988). Functional anatomy of
    macaque striate cortex. IV. Contrast and magno-parvo streams. The Journal
    of Neuroscience, 8(5), 1594–1609.
"""

import numpy as np
from ds_crfFit import crf_fit
from ds_pltAcrDpth import funcPltAcrDpth
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

print('-CRF fitting on data from Tootell et al. (1988)')


# ----------------------------------------------------------------------------
# *** Parameters

# Width of Gaussian (FWHM) for smoothing in order to simulate partial volume
# effect on depth profiles (for comparison with fMRI profiles). If zero, no
# smoothing is performed. Unit is the ratio of cortical depth, i.e. if 0.1,
# FWHM is 10% of cortical depth.
varSd = 0.1

# Number of x-values for which to solve the function when calculating model
# fit:
varNumX = 1000


# ----------------------------------------------------------------------------
# *** Preparations

# Stimulus contrast levels (independent variable):
vecEmpX = np.array([0.08, 0.18, 0.4, 1.0])

# Number of empirical contrast levels:
varNumConEmp = vecEmpX.shape[0]

# Tracer uptake (response, dependent variable):
lstEmpY = [np.array([70.0, 210.0, 560.0, 570.0]),  # Layer 3
           np.array([47.0, 285.0, 550.0, 680.0]),  # Layer 4A
           np.array([288.0, 341.0, 633.0, 582.0]),  # Layer 4B
           np.array([435.0, 461.0, 622.0, 768.0]),  # Layer 4Ca
           np.array([8.0, 117.0, 492.0, 819.0]),  # Layer 4Cb
           np.array([81.0, 239.0, 432.0, 475.0]),  # Layer 5
           np.array([455.0, 636.0, 859.0, 996.0])]  # Layer 6

# Relative position of layers:
lstLayPos = [670.0,  # Layer 3
             530.0,  # Layer 4A
             450.0,  # Layer 4B
             362.0,  # Layer 4Ca
             285.0,  # Layer 4Cb
             170.0,  # Layer 5
             55.0]  # Layer 6

# Get number of layers:
if np.greater(float(varSd), 0.0):
    # If smoothing is applied, the depth-profiles are upsampled before
    # smoothing. The plots are then created at the upsampled resolution.
    varNumLayers = int(max(lstLayPos))
else:
    varNumLayers = len(lstLayPos)

# Array for modelled y values, of the form aryMdlY[idxLayer, idxContrast]:
aryMdlY = np.zeros((varNumLayers, varNumX))

# Vector for response at 50% contrast:
vecHlfMax = np.zeros((varNumLayers))

# Vector for semisaturation contrast:
vecSemi = np.zeros((varNumLayers))

# Vector for model residuals, of the form aryRes[idxLayer, idxContrast]:
aryRes = np.zeros((varNumLayers, varNumConEmp))

# Vector for relative laminar depth of results (for plot):
vecLayPos = np.zeros((varNumLayers))


# ----------------------------------------------------------------------------
# *** Smoothing of depth profiles

# Smooth profiles in order to simulate fMRI partial volume effect:
if np.greater(float(varSd), 0.0):

    # Cortical depth is set to 1000, based on the reference frame of the
    # empirical values from the literature:
    varNumIntp = 1000.0

    # Position of original datapoints (before interpolation):
    vecPosOrig = np.asarray(lstLayPos)

    # Positions at which to sample (interpolate) depth profiles:
    vecPosIntp = np.linspace(0.0, varNumIntp, num=1000, endpoint=True)

    # Empirical data (from literature) from list to array:
    aryEmpY = np.asarray(lstEmpY)

    # Create function for interpolation:
    func_interp = interp1d(vecPosOrig,
                           aryEmpY,
                           kind='linear',
                           axis=0,
                           fill_value='extrapolate')

# interp1d - Possibly use other parameters?
# kind : str or int, optional
# Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’,
# ‘slinear’, ‘quadratic’, ‘cubic’ where ‘zero’, ‘slinear’, ‘quadratic’ and
# ‘cubic’ refer to a spline interpolation of zeroth, first, second or third
# order) or as an integer specifying the order of the spline interpolator to
# use. Default is ‘linear’.

    # Apply interpolation function:
    aryEmpIntY = func_interp(vecPosIntp)

    # Scale the standard deviation of the Gaussian kernel:
    varSdSc = np.float64(varNumIntp) * varSd

    # Smooth interpolated depth profiles:
    aryEmpSmthY = gaussian_filter1d(aryEmpIntY,
                                    varSdSc,
                                    axis=0,
                                    order=0,
                                    mode='nearest')

    # After interpolation, spacing of laminar positions is linear, with the
    # maximum value equal to the position of the outermost original datapoint:
    vecLayPos = np.linspace(0.0,
                            float(max(lstLayPos)),
                            num=int(max(lstLayPos)))


# ----------------------------------------------------------------------------
# *** Contrast response function fitting

# Loop through layers & fit function:
for idxLay in range(varNumLayers):

    print(('---Layer ' + str(idxLay + 1) + ' of ' + str(varNumLayers)))

    # The data structure is different if smoothing was applied:
    if np.greater(float(varSd), 0.0):
        vecEmpY = aryEmpSmthY[idxLay, :].reshape((1, varNumConEmp))
    else:
        # Reshape to fit expected function input:
        vecEmpY = lstEmpY[idxLay].reshape((1, varNumConEmp))

    # Rescale values:
    vecEmpY = np.multiply(vecEmpY, 0.001)

    aryMdlY[idxLay, :], vecHlfMax[idxLay], vecSemi[idxLay], \
        aryRes[idxLay, :] = crf_fit(vecEmpX, vecEmpY, strFunc='power',
                                    varNumX=varNumX)

    # Only fill vector with relative laminar positions if no smoothing is
    # performed (otherwise the spacing is linear):
    if not(np.greater(float(varSd), 0.0)):

        # Put relative laminar position of current layer into vector (for
        # plotting):
        vecLayPos[idxLay] = lstLayPos[idxLay]


# ----------------------------------------------------------------------------
# *** Prepare plots

# Rescale laminar position (to fit with plot settings):
vecLayPos = np.subtract(vecLayPos, 55.0)
vecLayPos = np.divide(vecLayPos, ((1000.0 - 55.0) / 0.85))
vecLayPos = np.add(vecLayPos, 0.1)

# Reshape arrays for plot:
vecSemi = vecSemi.reshape((1, varNumLayers))
vecHlfMax = vecHlfMax.reshape((1, varNumLayers))

# Dummy array for error bars:
aryError = np.zeros(vecSemi.shape)


# ----------------------------------------------------------------------------
# *** Plot results

# Path for figure:
strPath = '/Users/john/Desktop/Tootell_Smooth/semiTootell1988_smth_0p1.png'

# Plot semisaturation contrast:
funcPltAcrDpth(vecSemi, aryError, varNumLayers, 1, 80.0, 0.0, 0.5, True,
               [''], 'Cortical depth', 'Percent luminance contrast',
               'Semisaturation contrast', False, strPath, vecX=vecLayPos,
               varNumLblY=6, varXmin=0.0, varXmax=1.0)

# Path for figure:
strPath = '/Users/john/Desktop/Tootell_Smooth/hlfmaxTootell1988_smth_0p1.png'

# Plot response at 50% contrast:
funcPltAcrDpth(vecHlfMax, aryError, varNumLayers, 1, 80.0, 0.0, 1.0, False,
               [''], 'Cortical depth', 'Response [a.u.]',
               'Response at 50% contrast', False, strPath, vecX=vecLayPos,
               varNumLblY=6, varXmin=0.0, varXmax=1.0)
# ----------------------------------------------------------------------------
