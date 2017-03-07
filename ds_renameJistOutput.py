# -*- coding: utf-8 -*-
"""
Function of the depth sampling pipeline.

The purpose of this script is to rename output from the JIST process manager.
This is necessary because at the moment the depth-sampling pipeline can only
be run through the JIST GUI, and I did not manage to configure the pipeline
so that there would be a consistent naming of output files.

This script removes unwanted prefixes from the JIST output, separating the
file names at underscores.

This is a makeshift function that will hopefully become redundant once the
CBS depth sampling has a python interface.

@author: Ingo Marquardt, 17.02.2017
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


from os import listdir
from os import rename


# %% Define parameters

# Subject IDs & number of prefixes to remove for that subject:
dicSubId = {'20150930': 6,
            '20151118': 6,
            '20151127_01': 7,
            '20151130_02': 7,
            '20161205': 6,
            '20161207': 6,
            '20161212_02': 7,
            '20161214': 6,
            '20161219_01': 7,
            '20161219_02': 7}

# Paths with files to be renamed (subject ID left open):
lstPths = ['/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh/',
           '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh_era/stim_lvl_01/',  #noqa
           '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh_era/stim_lvl_02/',  #noqa
           '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh_era/stim_lvl_03/',  #noqa
           '/home/john/PhD/ParCon_Depth_Data/{}/cbs_distcor/lh_era/stim_lvl_04/']  #noqa

# File type (suffix):
strFleTpe = '.vtk'

# Character for parsing filenames:
strPrse = '_'

# %% Correct file names:

print('-Rename JIST output')

# Loop through subjects:
for strSub in dicSubId:

    print('---Processing: ' + strSub)

    # Number of prefixes to remove for this subject:
    varTmpPfxRmv = dicSubId[strSub]

    # Loop through directories:
    for strPth in lstPths:

        print('------Directory: ' + strPth.format(strSub))

        # List of all files and directories at target location:
        lstFiles = listdir(strPth.format(strSub))

        # Only keep files of specified type in the list:
        lstFiles = [i for i in lstFiles if i[-4:] == strFleTpe]

        # Loop through files:
        for strFle in lstFiles:

            # Parse file name:
            lstTmp = strFle.split(strPrse)[varTmpPfxRmv:]

            # The new file name:
            strTmp = '_'.join(lstTmp)

            # Rename the file:
            rename((strPth.format(strSub) + strFle),
                   (strPth.format(strSub) + strTmp))

print('-Done')
