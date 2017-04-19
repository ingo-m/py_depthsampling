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

# aryDpth = np.save('/home/john/Desktop/aryDpth.npy', aryHlfMax)
# aryDpth = np.load('/home/john/Desktop/aryDpth.npy')[0, :, :]
# varNumIntp=1000
# varSd=0.05

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
        dimensions, based on the first derivative of the depth profile along
        this dimension.
    varNumIntp : int
        Number of points at which to interpolate depth profiles before
        calculating the gradient.
    varSd : float
        Standard deviation of the Gaussian kernel used for smoothing, relative
        to cortical thickness (i.e. a value of 0.05 would result in a Gaussian
        with FWHM of 5 percent of the cortical thickness).

    Returns
    -------
    vecMin : np.array
        Relative position of the peak (between 0.0 and 1.0). 1D array with
        length equal to the number of depth profiles of the input array.

    Notes
    -----
    Depth profiles are upsampled and smoothed with a Gaussian kernel. The peak
    is defined as the maximum of the first derivative of the interpoalted,
    smoothed depth profile.
    
    Function of the depth sampling pipeline.
    """
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

    # Calculate second derivative along depth-dimension:
    aryDpthGrd = np.gradient(aryDpthSmth, axis=1)
    # aryDpthGrd = np.gradient(aryDpthGrd, axis=1)

    # Absolute of gradient:
    # aryDpthGrdAbs = np.absolute(aryDpthGrd)
    
    # Index of minimum:
    # vecPeak = np.argmin(aryDpthGrdAbs, axis=1)
    vecPeak = np.argmax(aryDpthGrd, axis=1)

    # Convert the indicies of minimum value into relative position (i.e.
    # relative cortical depth):
    vecPeak = np.divide(vecPeak, np.float64(varNumIntp))

    return vecPeak

#aa1 = aryDpth[7832, :]
#aa2 = aryDpthIntp[7832, :]
#aa3 = aryDpthSmth[7832, :]
#aa4 = aryDpthGrd[7832, :]
#aa5 = aryDpthGrdAbs[7832, :]
#aa6 = vecPeak[7832]
