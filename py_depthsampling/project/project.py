# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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
import pandas as pd
from py_depthsampling.get_data.load_csv_roi import load_csv_roi
# from py_depthsampling.get_data.load_vtk_single import load_vtk_single
from py_depthsampling.get_data.load_vtk_multi import load_vtk_multi
import seaborn as sns

strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/pRF_results_R2.vtk'
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/pRF_results_SD.vtk'
strPthX = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/pRF_results_x_pos.vtk'
strPthY = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/pRF_results_y_pos.vtk'

strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/feat_level_2_Pd_sst_cope.vtk'

strCsvRoi = '/home/john/PhD/GitHub/PacMan/analysis/20180118/08_depthsampling/rh/v1_mod.csv'

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






# --------------
# ***







def project():
    """
    Project parameter estimates into visual space based on pRF information.
    
    Parameters
    ----------

    Returns
    -------


    Notes
    -----

    """


    # Calculate polar angle map:
    aryPrfRes[:, :, :, 4] = np.arctan2(aryPrfRes[:, :, :, 1],
                                       aryPrfRes[:, :, :, 0])

    # Calculate eccentricity map (r = sqrt( x^2 + y^2 ) ):
    aryPrfRes[:, :, :, 5] = np.sqrt(np.add(np.power(aryPrfRes[:, :, :, 0],
                                                    2.0),
                                           np.power(aryPrfRes[:, :, :, 1],
                                                    2.0)))
    


varX = np.sin(varPol) * varEcc

varY = (np.cos((0.5 * np.pi) - varPol)) * varEcc

