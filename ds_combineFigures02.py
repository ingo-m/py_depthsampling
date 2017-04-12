"""
Combine SVG figures (depth plots).

Function of the depth sampling pipeline.

The purpose of this script is to combine figures (contrast response functions)
in SVG format into a summary figure.

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
strFigBase = '/home/john/PhD/Tex/contrast_response/combined_corrected/crf_V{}_dpth_{}.svg'  #noqa
lstPths = [strFigBase.format(str(var01 + 1), str(var02))
           for var01 in range(2) for var02 in range(5)]

# Output path:
strOt = '/home/john/PhD/Tex/contrast_response/combined_corrected/crf_summary.svg'  #noqa

# Create parent SVG figure:
objFigPrnt = sg.SVGFigure(width='1875.0 pix', height='600.0 pix')

# Load figures:
lstFigs = [None for i in range(len(lstPths))]
for idxIn in range(len(lstPths)):
    lstFigs[idxIn] = sg.fromfile(lstPths[idxIn])

# Retrieve plot objects:
lstPlts = [None for i in range(len(lstPths))]
for idxIn in range(len(lstPths)):
    lstPlts[idxIn] = lstFigs[idxIn].getroot()

# X positions for subplots:
varXstart = 10.0
varXsize = 370.0
varXstop = (len(lstPlts) * 0.5 * varXsize + varXstart)
aryPosX = np.arange(varXstart, varXstop, varXsize)
aryPosX = np.hstack((aryPosX, aryPosX))

# Y positions for subplots:
aryPosY = np.hstack((np.repeat([10.0], 5),
                     np.repeat([300.0], 5)))

# Move plots:
for idxIn in range(len(lstPths)):
    lstPlts[idxIn].moveto(aryPosX[idxIn], aryPosY[idxIn], scale=1.0)

# Append plots to parent figure object:
objFigPrnt.append(lstPlts)

# Save new parent SVG file:
objFigPrnt.save(strOt)
