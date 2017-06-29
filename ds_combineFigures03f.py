"""
Combine SVG figures (depth plots).

Function of the depth sampling pipeline.

The purpose of this script is to combine figures (CRF data based on Tootell et
al., 1988) in SVG format into a summary figure.

This script is somewhat of a make shift solution to combine figures, tailored
for one particular use case (better than combining them manually and having to
repeat that every time the data change).

This script uses the svgutils module, thanks to
Bartosz Telenczuk @ https://github.com/btel

To make svgutils available:
```
pip install svgutils
```
"""

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
import svgutils.transform as sg

# Paths to figures:
strPthBase = '/home/john/PhD/Tex/Tootell_1988'
lstPths = ['/hlfmaxTootell1988.svg',
           '/semiTootell1988.svg']
lstPths = [(strPthBase + x) for x in lstPths]

# Output path:
strOt = '/home/john/Dropbox/ParCon_Manuscript/Figures_Source/Figure_09_Tootell_1988.svg'  #noqa

# Create parent SVG figure:
objFigPrnt = sg.SVGFigure(width='1680.0 pix', height='750.0 pix')

# Load figures:
lstFigs = [None for i in range(len(lstPths))]
for idxIn in range(len(lstPths)):
    lstFigs[idxIn] = sg.fromfile(lstPths[idxIn])

# Retrieve plot objects:
lstPlts = [None for i in range(len(lstPths))]
for idxIn in range(len(lstPths)):
    lstPlts[idxIn] = lstFigs[idxIn].getroot()

# X positions for subplots:
lstPosX = [10.0, 820.0]

# Y positions for subplots:
lstPosY = [10.0, 10.0]

# Scaling factors for subplots:
lstScale = [1.0, 1.0]

# Move plots:
for idxIn in range(len(lstPths)):
    lstPlts[idxIn].moveto(lstPosX[idxIn],
                          lstPosY[idxIn],
                          scale=lstScale[idxIn])

# Append plots to parent figure object:
objFigPrnt.append(lstPlts)

# Save new parent SVG file:
objFigPrnt.save(strOt)
