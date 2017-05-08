# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema


def find_peak(aryDpth, varNumIntp=100, varSd=0.05):
    """
    Find peak in cortical depth profile.

    Parameters
    ----------
    aryDpth : np.array
        Cortical depth profiles. 2D array where the first dimension
        corresponds to the number of depth profiles (e.g. bootstrapped
        versions of the depth profile) and the second dimension corresponds to
        cortical depth. The peak will be searched for along the second
        dimensions.
    varNumIntp : int
        Number of points at which to interpolate depth profiles before
        searching for a peak.
    varSd : float
        Standard deviation of the Gaussian kernel used for smoothing, relative
        to cortical thickness (i.e. a value of 0.05 would result in a Gaussian
        with FWHM of 5 percent of the cortical thickness).

    Returns
    -------
    vecMin : np.array
        Relative position of the peak (between 0.0 and 1.0) for each depth
        profile. 1D array with length equal to the number of depth profiles of
        the input array.

    Notes
    -----
    Depth profiles are upsampled and smoothed with a Gaussian kernel. The
    function scipy.signal.argrelextrema is used to search for a peak.

    Function of the depth sampling pipeline.
    """
    # Number of depth profiles (e.g. resampling iterations):
    varNumIt = aryDpth.shape[0]

    # Identify number of depth levels:
    varNumDpth = aryDpth.shape[1]

    # Position of original datapoints (before interpolation):
    vecPosOrig = np.linspace(0, 1.0, num=varNumDpth, endpoint=True)

    # Positions at which to sample (interpolate) depth profiles:
    vecPosIntp = np.linspace(0, 1.0, num=varNumIntp, endpoint=True)

    # Create function for interpolation:
    func_interp = interp1d(vecPosOrig,
                           aryDpth,
                           kind='linear',
                           axis=1,
                           fill_value='extrapolate')

    # Apply interpolation function:
    aryDpthIntp = func_interp(vecPosIntp)

    # Scale the standard deviation of the Gaussian kernel:
    varSdSc = np.float64(varNumIntp) * varSd

    # Smooth interpolated depth profiles:
    aryDpthSmth = gaussian_filter1d(aryDpthIntp,
                                    varSdSc,
                                    axis=1,
                                    order=0,
                                    mode='nearest')

    # Order of the search for the local maximum: how many points on each side
    # to use for the comparison to consider.
    varNumOrd = int(np.around((np.float64(varNumIntp) * 0.01),
                              decimals=0))

    # Identify peaks (the algorithm procudes a tuple of two arrays, the first
    # with the indicies of cases (i.e. bootstrap iterations) for which a peak
    # was identified, the second with the indicies of the peak.
    vecPos, vecPeak = argrelextrema(aryDpthSmth,
                                    np.greater,
                                    axis=1,
                                    order=varNumOrd,
                                    mode='clip')

    # Number of cases for which a peak was found:
    varNumPeaks = vecPeak.shape[0]

    print(('------Identified local maxima in '
           + str(varNumPeaks)
           + ' out of '
           + str(varNumIt)
           + ' cases.'))

    # Create vector with one peak position per case (for depth profiles that
    # are monotonically increasing, no local maximum can be identified). There
    # peak it their maximum value.
    vecPeak2 = np.zeros((varNumIt), dtype=np.int64)
    vecPeak2[vecPos] = vecPeak

    # Cases for which no loacl maximum has been identified (depth profiles with
    # monotonic increase):
    vecPosMono = np.equal(vecPeak2, 0)

    # Identifty position of global maximum:
    vecMax = np.argmax(aryDpthSmth, axis=1)

    # Replace zeros with global maximum for respective cases:
    vecPeak2[vecPosMono] = vecMax[vecPosMono]

    # Number of cases for which a local or global maxima was found:
    varNumPeaks2 = vecPeak2.shape[0]

    print(('------Identified local or global maxima in '
           + str(varNumPeaks2)
           + ' out of '
           + str(varNumIt)
           + ' cases.'))

    # Convert the indicies of peak value into relative position (i.e.
    # relative cortical depth):
    vecPeak2 = np.divide(vecPeak2, np.float64(varNumIntp))

    return vecPeak2
