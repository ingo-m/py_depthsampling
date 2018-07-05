# -*- coding: utf-8 -*-
"""Calculate similarity between visual field projections and stimulus model."""

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
from scipy.ndimage.filters import gaussian_filter


# -----------------------------------------------------------------------------
# *** Define parameters



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

# Dimensions of visual space model (x & y):
tplSzeVslMdl = (200, 200)

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Prepare stimulus masks

# Vector with visual space x-coordinates:
vecVslX = np.linspace(varExtXmin, varExtXmax, num=tplSzeVslMdl[0],
                      endpoint=True)

# Vector with visual space y-coordinates:
vecVslY = np.linspace(varExtYmin, varExtYmax, num=tplSzeVslMdl[1],
                      endpoint=True)

# Eccentricity map of visual space:
aryEcc = np.sqrt(
                 np.add(
                        np.power(vecVslX[:, None], 2.0),
                        np.power(vecVslY[None, :], 2.0)
                        )
                 )

# Polar angle map of visual space:
aryPol = np.arctan2(vecVslY[:, None], vecVslX[None, :])

# Array representing PacMan shape (binary mask):
aryPacMan = np.multiply(
                        np.less_equal(aryEcc, 3.75),
                        np.logical_or(
                                      np.less(aryPol, np.deg2rad(-35.0)),
                                      np.greater(aryPol, np.deg2rad(35.0))
                                      )
                        ).astype(np.float64)

# Get binary mask of PacMan edge using gradient of PacMan:
lstGrd = np.gradient(aryPacMan.astype(np.float64))
aryEdge = np.greater(
                     np.add(
                            np.absolute(lstGrd[0]),
                            np.absolute(lstGrd[1])
                            ),
                     0.0).astype(np.float64)

# Scale width of edge mask (so that it becomes less dependent on resolution of
# visual field model):
varEdgeWidth = 0.01
varSd = varEdgeWidth * float(tplSzeVslMdl[0])
aryEdge = gaussian_filter(aryEdge, varSd, order=0, mode='nearest',
                          truncate=4.0)
aryEdge = np.greater(aryEdge, 0.075).astype(np.float64)

# Visual space outside of stimulus:
aryOuter = np.less(np.add(aryPacMan, aryEdge), 0.1)
# -----------------------------------------------------------------------------


