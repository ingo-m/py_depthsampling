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


def find_peak(aryDpth, varNumIntp=100, varSd=0.05, lgcStat=True, varThr=None):
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
    lgcStat : bool, optional
        Whether to print status messages.
    varThr : float or None, optional
        Amplitude threshold for peak identification. For example, if `varThr =
        0.05`, peaks with an absolute amplitude that is greater than the mean
        amplitude (over cortical depth) plus 0.05 are identified. The rational
        for this is that even for very flat profiles a peak is identified. The
        threshold does not influence the peak search; instead, if a threshold
        is provided, an additional output vector is returned, containing
        boolean values (true if peak amplitude exceeds threshold).

    Returns
    -------
    vecMin : np.array
        Relative position of the peak (between 0.0 and 1.0) for each depth
        profile. 1D array with length equal to the number of depth profiles of
        the input array.
    vecLgc : np.array, optional
        Peak amplitude threshold check. 1D vector with boolean values (true if
        peak amplitude exceeds threshold). One value per depth profile.
        Optional; only returned if a peak amplitude threshold is provided.

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
    varNumOrd = int(np.around((np.float64(varNumIntp) * 0.1),
                              decimals=0))

    # Identify peaks (the algorithm procudes a tuple of two arrays, the first
    # with the indicies of cases (i.e. bootstrap iterations) for which a peak
    # was identified, the second with the indicies of the peak.
    vecIdx, vecPeak = argrelextrema(aryDpthSmth,
                                    np.greater,
                                    axis=1,
                                    order=varNumOrd,
                                    mode='clip')

    # Create vector with one peak position per case. In some cases, no local
    # maximum could be identified. For those cases, the global maximum will
    # be selected as peak. In other cases, there may be more than one local
    # maximum. In this case, the local maximum with the greater amplitude will
    # be selected.
    vecPeak2 = np.zeros((varNumIt), dtype=np.int64)

    # Vector for peak amplitudes:
    vecAmp = np.zeros((varNumIt))

    # Loop through all identified peaks (there can be more than one per depth
    # profile, or none):
    for idxPeak in range(vecPeak.shape[0]):

        # Index of current peak (i.e., which iteration does the current peak
        # refer to with respect to aryDpthSmth[iteration, depthlevel]):
        varIdxPeak01 = vecIdx[idxPeak]

        # Amplitude of current peak:
        varAmplt01 = aryDpthSmth[varIdxPeak01, vecPeak[idxPeak]]

        # Do not compare to previous peak on first iteration:
        if np.greater(idxPeak, 0):

            # Does the current peak refer to the same iteration (with respect
            # to aryDpthSmth[iteration, depthlevel])?
            if varIdxPeak02 == varIdxPeak01:

                # Is the amplitude of the current peak greater than that of
                # the previous peak?
                if np.greater(varAmplt01, varAmplt02):

                    # Replace peak position in output array (relative peak
                    # position with respect to cortical depth):
                    vecPeak2[varIdxPeak01] = vecPeak[idxPeak]

                    # Replace old amplitude with new (greater) amplitude:
                    vecAmp[varIdxPeak01] = varAmplt01

            else:

                # New iteration (with respect to aryDpthSmth[iteration,
                # depthlevel], place relative peak position in output array:
                vecPeak2[varIdxPeak01] = vecPeak[idxPeak]

                # Save amplitude:
                vecAmp[varIdxPeak01] = varAmplt01

        else:

            # First iteration:
            vecPeak2[varIdxPeak01] = vecPeak[idxPeak]
            vecAmp[varIdxPeak01] = varAmplt01

        # Remember index of current peak (i.e. which depth profile does it
        # refer to, so as to compare on the next iteration of the loop whether
        # it refers to the same depth profile):
        varIdxPeak02 = vecIdx[idxPeak]

        # Remember amplitude of current peak (for comparison on next iteration
        # of the loop):
        varAmplt02 = aryDpthSmth[varIdxPeak02, vecPeak[idxPeak]]

    # Number of cases for which a peak was found:
    varNumPeaks = np.shape(np.unique(vecIdx))[0]

    if lgcStat:
        print(('------Identified local maxima in '
               + str(varNumPeaks)
               + ' out of '
               + str(varNumIt)
               + ' cases.'))

    # Cases for which no local maximum has been identified (depth profiles with
    # monotonic increase). In this case, the global maximum is defined as the
    # peak.
    vecPosMono = np.equal(vecPeak2, 0)

    # Identifty position of global maximum:
    vecArgMax = np.argmax(aryDpthSmth, axis=1)

    # Replace zeros with global maximum for respective cases:
    vecPeak2[vecPosMono] = vecArgMax[vecPosMono]

    # Amplitude of global maximum:
    vecMax = np.max(aryDpthSmth, axis=1)

    # Save amplitude of global maximum for cases where no local maximum was
    # found:
    vecAmp[vecPosMono] = vecMax[vecPosMono]

    # Number of cases for which a local or global maxima was found:
    varNumPeaks2 = vecPeak2.shape[0]

    if lgcStat:
        print(('------Identified local or global maxima in '
               + str(varNumPeaks2)
               + ' out of '
               + str(varNumIt)
               + ' cases.'))

    # Convert the indicies of peak value into relative position (i.e.
    # relative cortical depth):
    vecPeak2 = np.divide(vecPeak2, np.float64(varNumIntp))

    # If no peak threshold is given, just return the vector with relative peak
    # positions.
    if varThr is None:

        return vecPeak2

    else:

        # Mean along depth dimension (separately for each depth profile):
        vecMne = np.mean(aryDpth, axis=1)

        # Add peak-threshold to mean:
        vecThr = np.add(np.absolute(vecMne), varThr)

        # Is the absolute amplitude greater than threshold?
        vecLgc = np.greater(np.absolute(vecAmp), vecThr)

        return vecPeak2, vecLgc
