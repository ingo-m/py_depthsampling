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

/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/eccentricity.vtk
/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/Pd_pe1.vtk
/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/polar_angle.vtk
/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/20180118/cbs/rh/SD.vtk

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

