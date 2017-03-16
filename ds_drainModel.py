# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

The purpose of this script is to remove the contribution of lower cortical
depth levels to the signal at each consecutive depth level. In other words,
at a given depth level, the contribution from lower depth levels is removed
based on the model proposed by Markuerkiaga et al. (2016).

The following data from Markuerkiaga et al. (2016) is used in this script:

    'The cortical layer boundaries of human V1 in the model were fixed
    following de Sousa et al. (2010) and Burkhalter and Bernardo (1989):
    layer VI, 20%;
    layer V, 10%;
    layer IV, 40%;
    layer II/III, 20%;
    layer I, 10%
    (values rounded to the closest multiple of 10).' (p. 492)

Moreover, the absolute (?) signal contribution in each layer for a GE sequence
as depicted in Figure 3F (p. 495):

    Layer VI:
        1.9 * var6
    Layer V:
        (1.5 * var 5)
        + ((2.1 - 1.5) * var6)
    Layer IV:
        (2.2 * var4)
        + ((2.5 - 2.2) * var5)
        + ((3.1 - 2.5) * var6)
    Layer II/III:
        (1.7 * var23)
        + ((3.0 - 1.7) * var4)
        + ((3.3 - 3.0) * var5)
        + ((3.8 - 3.3) * var6)
    Layer I:
        (1.6* var1)
        + ((2.3 - 1.6) * var23)
        + ((3.6 - 2.3) * var4)
        + ((3.9 - 3.6) * var5)
        + ((4.4 - 3.9) * var6)

These values are translated into the corrected relative contribution of local
signal (originating from the given depth level) and signal from lower layers:

    Layer VI:
    varCrct6 = varEmp6 / 1.9
    
    Layer V:
    varCrct5 = (varEmp5
                - (0.6 * varCrct6)) / 1.5
    
    Layer IV:
    varCrct4 = (varEmp4
                - (0.3 * varCrct5)
                - (0.6 * varCrct6)) / 2.2
    
    Layer II/III:
    varCrct23 = (varEmp23
                 - (1.3 * varCrct4)
                 - (0.3 * varCrct5)
                 - (0.5 * varCrct6)) / 1.7
    
    Layer I:
    varCrct1 = (varEmp1
                - (0.7 * varCrct23)
                - (1.3 * varCrct4)
                - (0.3 * varCrct5)
                - (0.5 * varCrct6)) / 1.6


Reference:
Markuerkiaga, I., Barth, M., & Norris, D. G. (2016). A cortical vascular model
    for examining the specificity of the laminar BOLD signal. Neuroimage, 132,
    491-498.

@author: Ingo Marquardt, 16.03.2017
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

#varEmp6 = 1.9
#varEmp5 = 2.1
#varEmp4 = 3.1
#varEmp23 = 3.8
#varEmp1 = 4.4

#varConstant = 1.0
#varEmp6 = 1.9 * varConstant
#varEmp5 = 2.1 * varConstant
#varEmp4 = 3.1 * varConstant + 2.0
#varEmp23 = 3.8 * varConstant + 1.5
#varEmp1 = 4.4 * varConstant + 1.0

varEmp6 = 1.0
varEmp5 = 1.5
varEmp4 = 2.0
varEmp23 = 2.2
varEmp1 = 2.4

#Layer VI:
varCrct6 = varEmp6 / 1.9

#Layer V:
varCrct5 = (varEmp5
            - (0.6 * varCrct6)) / 1.5

# Layer IV:
varCrct4 = (varEmp4
            - (0.3 * varCrct5)
            - (0.6 * varCrct6)) / 2.2

# Layer II/III:
varCrct23 = (varEmp23
             - (1.3 * varCrct4)
             - (0.3 * varCrct5)
             - (0.5 * varCrct6)) / 1.7

# Layer I:
varCrct1 = (varEmp1
            - (0.7 * varCrct23)
            - (1.3 * varCrct4)
            - (0.3 * varCrct5)
            - (0.5 * varCrct6)) / 1.6

vecEmp = np.array([varEmp6, varEmp5, varEmp4, varEmp23, varEmp1])
vecCrct = np.array([varCrct6, varCrct5, varCrct4, varCrct23, varCrct1])
aryComb = np.array([vecEmp, vecCrct]).T
