#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate difference between visual field projections.
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


import numpy as np
from py_depthsampling.project.plot import plot


# -----------------------------------------------------------------------------
# ### Parameters

# ROI:
lstRoi = ['v1', 'v2']

# Parent path of visual field projections (ROI and condition name left open):
strPthPrnt = '/home/john/Dropbox/Kanizsa_Depth_Data/Higher_Level_Analysis/project/{}_feat_level_2_{}_sst_pe_allGM.npy'

# Names of visual field projection condition (first VFP minus second VFP).
lstPth01 = ['kanizsa_flicker',
            'rotated_flicker',
            'kanizsa_flicker',
            'kanizsa_static']
lstPth02 = ['kanizsa_static',
            'rotated_static',
            'rotated_flicker',
            'rotated_static']

# Figure output path (ROI and conditions left open):
strPathOut = '/home/john/Dropbox/Kanizsa_Project/Plots/project_diff/{}_{}_minus_{}.png'

# Extent of visual space (for axes labels):
varExtXmin = -8.3
varExtXmax = 8.3
varExtYmin = -5.19
varExtYmax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ### Create VFP difference plot

# Loop through ROIs:
for strRoi  in lstRoi:

    # Loop through comparisons:
    for idxCom in range(len(lstPth01)):

        # Path of VFP files:
        strPth01 = strPthPrnt.format(strRoi, lstPth01[idxCom])
        strPth02 = strPthPrnt.format(strRoi, lstPth02[idxCom])

        # Figure title:
        strTtl = (lstPth01[idxCom]
                  + ' minus '
                  + lstPth02[idxCom])

        # Output file path:
        strPathOutTmp = strPathOut.format(strRoi,
                                          lstPth01[idxCom],
                                          lstPth02[idxCom])

        # Load visual field projections from disk:
        ary01 = np.load(strPth01)
        ary02 = np.load(strPth02)

        # Subtract visual field projections:
        aryDiff = np.subtract(ary01, ary02)

        plot(aryDiff,
             strTtl,
             'x-position',
             'y-position',
             strPathOutTmp,
             tpleLimX=(varExtXmin, varExtXmax, 3.0),
             tpleLimY=(varExtYmin, varExtYmax, 3.0),
             varMin=-2.0,
             varMax=2.0)
# -----------------------------------------------------------------------------
