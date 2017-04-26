"""
Combine SVG figures (depth plots).

Function of the depth sampling pipeline.

The purpose of this script is to combine figures (depth profile normalisation
demo) in SVG format into a summary figure.

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

import svgutils.transform as sg

# ----------------------------------------------------------------------------
# *** Define parameters

# Paths to figures:
strPthBase = '/home/john/PhD/Tex/dpth_norm_demo/plots/plt_'
lstPths = [(strPthBase + str(x).zfill(2) + '.svg') for x in range(1, 21)]

# Output path:
strOt = '/home/john/PhD/Tex/dpth_norm_demo/plots/summary.svg'  #noqa

# Create parent SVG figure:
objFigPrnt = sg.SVGFigure(width='2000.0 pix', height='2000.0 pix')

# Load figures:
lstFigs = [None for i in range(len(lstPths))]
for idxPlt in range(len(lstPths)):
    lstFigs[idxPlt] = sg.fromfile(lstPths[idxPlt])

# Retrieve plot objects:
lstPlts = [None for i in range(len(lstPths))]
for idxPlt in range(len(lstPths)):
    lstPlts[idxPlt] = lstFigs[idxPlt].getroot()

# x size of element plots:
varEleSzeX = 506.0
# y size of element plots:
varEleSzeY = 450.0
# x distance between element plots:
varEleDistX = 10.0
# y distance between element plots:
varEleDistY = 50.0

# y size of combined plots:
varCombSzeY = 675.0
# y distance between combined plots:
varCombDistY = 0.0

# Scaling factor for combined plots:
varSclComb = 1.3

# ----------------------------------------------------------------------------
# *** X-positions

# X positions for subplots:
lstPosX = [None] * len(lstPlts)

varPosTmp = 0.0

# For the first 12 plots every third plot is a combined plot with a deviating
# x-position, we use a counter & if condition to handle this:
varCnt = 0

# Loop through first 12 plots:
for idxPlt in range(0, 12):
    # If this is a linear & sinusoidal element plot (i.e. it is not a combined
    # plot):
    if varCnt < 2:
        varPosTmp += varEleDistX
        lstPosX[idxPlt] = varPosTmp
        varPosTmp += varEleSzeX
        varCnt += 1
    else:
        # If this is a combined plot, its x-position is equal to that of the
        # second-previous element plot:
        lstPosX[idxPlt] = lstPosX[idxPlt - 2]
        varCnt = 0

# Loop through remaining plots (profiles normalisation by subtraction):
varPosTmp = 0
for idxPlt in range(12, 16):
    varPosTmp += varEleDistX
    lstPosX[idxPlt] = varPosTmp
    varPosTmp += (1.0 * varEleSzeX + 1.0 * varEleSzeX)

# Loop through remaining plots (profiles normalisation by division):
varPosTmp = 0
for idxPlt in range(16, 20):
    varPosTmp += varEleDistX
    lstPosX[idxPlt] = varPosTmp
    varPosTmp += (1.0 * varEleSzeX + 1.0 * varEleSzeX)

# ----------------------------------------------------------------------------
# *** Y-positions

# Y positions for subplots:
lstPosY = [None] * len(lstPlts)

# For the first 12 plots every third plot is a combined plot with a deviating
# y-position, we use a counter & if condition to handle this:
varCnt = 0

# Loop through first 12 plots:
for idxPlt in range(0, 12):
    # If this is a linear & sinusoidal element plot (i.e. it is not a combined
    # plot):
    if varCnt < 2:
        lstPosY[idxPlt] = (0.0 * varEleSzeY
                           + 1.0 * varEleDistY
                           + 0.0 * varCombSzeY
                           + 0.0 * varCombDistY)
        varCnt += 1
    else:
        # If this is a combined plot:
        lstPosY[idxPlt] = (1.0 * varEleSzeY
                           + 1.0 * varEleDistY
                           + 0.0 * varCombSzeY
                           + 1.0 * varCombDistY)
        varCnt = 0

# Loop through remaining plots (profiles normalisation by subtraction):
for idxPlt in range(12, 16):
    lstPosY[idxPlt] = (1.0 * varEleSzeY
                       + 1.0 * varEleDistY
                       + 1.0 * varCombSzeY
                       + 2.0 * varCombDistY)

# Loop through remaining plots (profiles normalisation by division):
varPosTmp = 0
for idxPlt in range(16, 20):
    lstPosY[idxPlt] = (1.0 * varEleSzeY
                       + 1.0 * varEleDistY
                       + 2.0 * varCombSzeY
                       + 3.0 * varCombDistY)

# ----------------------------------------------------------------------------
# *** Scaling factors

# Scaling factors for subplots:
lstScale = [None] * len(lstPlts)

varCnt = 0

# Loop through first 12 plots:
for idxPlt in range(0, 12):
    # If this is a linear & sinusoidal element plot (i.e. it is not a combined
    # plot):
    if varCnt < 2:
        lstScale[idxPlt] = 1.0
        varCnt += 1
    else:
        # If this is a combined plot:
        lstScale[idxPlt] = varSclComb
        varCnt = 0

# Loop through remaining plots (profiles normalisation by subtraction):
for idxPlt in range(12, 16):
    lstScale[idxPlt] = varSclComb

# Loop through remaining plots (profiles normalisation by division):
varPosTmp = 0
for idxPlt in range(16, 20):
    lstScale[idxPlt] = varSclComb


# ----------------------------------------------------------------------------
# *** Arrange plots

# Move plots:
for idxPlt in range(len(lstPths)):
    lstPlts[idxPlt].moveto(lstPosX[idxPlt],
                           lstPosY[idxPlt],
                           scale=lstScale[idxPlt])


# ----------------------------------------------------------------------------
# *** Add column & row text labels

strTxt01 = 'Linear & sinusiodal term additive'
objTxt01 = sg.TextElement(20,
                          20,
                          strTxt01,
                          size=80,  # font="Verdana",  # DejaVuSans
                          weight="bold")


strTxt02 = 'Linear & sinusiodal term multiplicative'
strTxt03 = 'Linear additive, sinusoidal multiplicative'
strTxt04 = 'Linear multiplicative, sinusoidal additive'


# ----------------------------------------------------------------------------
# *** Move redundant x-axis labels

# List of figures with redundant x-axis labels:
lstRedX = [3, 6, 9, 12, 13, 14, 15, 16]

# Move redundant axis labels to the side:
for idxPlt in lstRedX:
    # Index of current plot:
    varTmp = idxPlt - 1
    # Handle to current plot x-axis:
    objTmp = lstPlts[varTmp].find_id('text_3')
    objTmp.moveto(5000.0, 0.0, scale=1)


# ----------------------------------------------------------------------------
# *** Move redundant y-axis labels

# For each figure, whether or not to remove y-axis labels:
lstRedY = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
# lstRedY = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20]

# The IDs of the redundant y-labels are not the same for all figures. List of
# IDs of respective axis label for all figures (also for those for which it
# will not be removed):
lstRedLblY = [7, 7, 9, 7, 7, 9, 7, 7, 9, 7, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8]

# Move redundant axis labels to the side:
for idxPlt in range(len(lstPlts)):
    if lstRedY[idxPlt] == 1:
        # ID of current plot:
        strTmp = ('text_' + str(lstRedLblY[idxPlt]))
        # Handle to current plot x-axis:
        objTmp = lstPlts[idxPlt].find_id(strTmp)
        objTmp.moveto(5000.0, 0.0, scale=1)


# ----------------------------------------------------------------------------
# *** Save figure

# Create parent SVG figure:
objFigPrnt = sg.SVGFigure(width='4150.0 pix', height='2600.0 pix')

# Append plots to parent figure object:
objFigPrnt.append(lstPlts)

# Append text to parent figure object:
objFigPrnt.append([objTxt01])

# Save new parent SVG file:
objFigPrnt.save(strOt)
# ----------------------------------------------------------------------------
