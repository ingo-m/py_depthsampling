# -*- coding: utf-8 -*-
"""Plot parameter estimates vs. eccentricity and find point spread function."""

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

# -----------------------------------------------------------------------------
# *** Define parameters

# Load projection from (ROI, condition, depth level label left open):
strPthNpy = '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/project/{}_{}_{}.npy'  #noqa

# Depth level condition labels (to complete input file names):
lstDpthLbl = ['allGM', 'deepGM', 'midGM', 'superficialGM']

# ROI ('v1', 'v2', or 'v3'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI, condition, depth level label left open):
strPthPltOt = '/home/john/Dropbox/PacMan_Plots/project/psf_pe/{}_{}_{}'  #noqa

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
          'Pd_trn', 'Cd_trn', 'Ps_trn',
          'Pd_min_Ps_sst', 'Pd_min_Cd_sst', 'Cd_min_Ps_sst',
          'Pd_min_Cd_Ps_sst',
          'Pd_min_Cd_Ps_trn',
          'Pd_min_Ps_trn', 'Pd_min_Cd_trn', 'Cd_min_Ps_trn']

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Project parametric map into visual space')

# Loop through depth levels, ROIs, and conditions:
for idxDpth in range(len(lstDpthLbl)):  #noqa
    for idxRoi in range(len(lstRoi)):
        for idxCon in range(len(lstCon)):

idxDpth = 0
idxRoi = 0
idxCon = 2

            # File name of npy file for current condition:
            strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                            lstCon[idxCon],
                                            lstDpthLbl[idxDpth])

            # Load visual field projection:
            aryVslSpc = np.load(strPthNpyTmp)

            # Size of visual field projection (in x- and y-directions):
            varNumX = aryVslSpc.shape[0]
            varNumY = aryVslSpc.shape[1]

            # Vector with visual space coordinates of elements in `aryVslSpc`:
            vecCorX = np.linspace(varExtXmin, varExtXmax, num=varNumX,
                                  endpoint=True)
            vecCorY = np.linspace(varExtYmin, varExtYmax, num=varNumY,
                                  endpoint=True)

            # Complete visual field arrays with x- and y-positions. Note that
            # the x-coordinates are increasing along the 0th axis, and the
            # y-coordinates are increasing along the 1st axis.
            aryCorX = np.repeat(vecCorX[:, None], varNumX, axis=1)
            aryCorY = np.repeat(vecCorY[None, :], varNumY, axis=0)

            # Calculate eccentricity:
            aryEcc = np.sqrt(
                             np.add(
                                    np.power(aryCorX, 2.0),
                                    np.power(aryCorY, 2.0)
                                    )
                             )




            # Select left visual field:
            aryVslSpc = aryVslSpc[np.less_equal(vecCorX, 0.0), :]
            aryEcc = aryEcc[np.less_equal(vecCorX, 0.0), :]

            # Flatten arrays:
            vecVslSpc = aryVslSpc.flatten()
            vecEcc = aryEcc.flatten()

import matplotlib
import matplotlib.pyplot as plt


            varSizeX = 1500.0
            varSizeY = 1500.0

    


            # Create figure:
            fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                        (varSizeY * 0.5) / varDpi),
                               dpi=varDpi)
        
            # Create axis:
            axs01 = fgr01.add_subplot(111)


plt.scatter(vecEcc, vecVslSpc, s=0.2, marker=".")


            plt01 = axs01.plot(vecEcc,
                               vecVslSpc,
                               linewidth=1.0,
                               antialiased=True)



aaa = np.exp((np.add(aryVslSpc, (np.abs(np.min(aryVslSpc)) + 0.01))))



aryLgc = np.multiply(np.greater_equal(aryEcc, 3.72),
                     np.less_equal(aryEcc, 3.78))

aryEcc[aryLgc] = 0.0

aaa = aryCorX.T
aaa = (np.fliplr(aryCorY)).T

# -----------------------------------------------------------------------------
