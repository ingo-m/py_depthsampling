"""
Combine SVG figures (depth plots).

Function of the depth sampling pipeline.

The purpose of this script is to combine figures (containing depth plots) in
SVG format pertaining to different ROIs/processing stages into a summary
figure.

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

# Paths to figures:
strFig01 = '/home/john/PhD/Tex/deconv/v1_model_1/deconv_v1_m1_beforeacrsSubsMeanShading.svg'
strFig02 = '/home/john/PhD/Tex/deconv/v1_model_1/deconv_v1_m1_afteracrsSubsMeanShading.svg'
strFig03 = '/home/john/PhD/Tex/deconv/v2_model_1/deconv_v2_m1_beforeacrsSubsMeanShading.svg'
strFig04 = '/home/john/PhD/Tex/deconv/v2_model_1/deconv_v2_m1_afteracrsSubsMeanShading.svg'

# Output path:
strOt = '/home/john/PhD/Tex/deconv/summary.svg'

# Create parent SVG figure:
objFigPrnt = sg.SVGFigure(width='1950.0 pix', height='1450.0 pix')

# Load figures:
objFig01 = sg.fromfile(strFig01)
objFig02 = sg.fromfile(strFig02)
objFig03 = sg.fromfile(strFig03)
objFig04 = sg.fromfile(strFig04)

# Retrieve plot objects:
objPlt01 = objFig01.getroot()
objPlt02 = objFig02.getroot()
objPlt03 = objFig03.getroot()
objPlt04 = objFig04.getroot()

# Move plots:
objPlt01.moveto(10.0, 1.0, scale=1.0)
objPlt02.moveto(880.0, 1.0, scale=1.0)
objPlt03.moveto(10.0, 700.0, scale=1.0)
objPlt04.moveto(880.0, 700.0, scale=1.0)

# Move redundant axis labels & legends:
objTmp = objPlt01.find_id('text_3')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt02.find_id('text_9')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt02.find_id('text_3')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt04.find_id('text_9')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt01.find_id('legend_1')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt03.find_id('legend_1')
objTmp.moveto(2000.0, 2000.0, scale=1)

objTmp = objPlt04.find_id('legend_1')
objTmp.moveto(2000.0, 2000.0, scale=1)

# Append plots to parent figure object:
objFigPrnt.append([objPlt01, objPlt02, objPlt03, objPlt04])

# Save new parent SVG file:
objFigPrnt.save(strOt)
