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

# Paths of visual field projections (second VFP will be subtracted from first
# VFP).
strPth01 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/project/v2_feat_level_2_kanizsa_rotated_sst_pe_allGM.npy'
strPth02 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/project/v2_feat_level_2_kanizsa_sst_pe_allGM.npy'

# Figure title:
strTtl = 'Kanizsa - Kanizsa rotated'

# Figure output path:
strPathOut = '/home/john/Desktop/v2_project_diff_pe.png'

# Extent of visual space (for axes labels):
varExtXmin = -8.3
varExtXmax = 8.3
varExtYmin = -5.19
varExtYmax = 5.19
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ### Create VFP difference plot

# Load visual field projections from disk:
ary01 = np.load(strPth01)
ary02 = np.load(strPth02)

# Subtract visual field projections:
aryDiff = np.subtract(ary02, ary01)

plot(aryDiff,
     strTtl,
     'x-position',
     'y-position',
     strPathOut,
     tpleLimX=(varExtXmin, varExtXmax, 3.0),
     tpleLimY=(varExtYmin, varExtYmax, 3.0),
     varMin=None,
     varMax=None)
# -----------------------------------------------------------------------------
