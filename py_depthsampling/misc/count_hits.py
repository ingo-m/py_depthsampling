# -*- coding: utf-8 -*-
"""Function of the depth sampling pipeline."""

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

import csv
import re


def count_hits(lstPath,
               strReg="You have detected [0-9]+ out of [0-9]+ targets."):
    """
    Read number of successful trials from log file.
    
    Parameters
    ----------
    lstPath : list
        List of paths to log files to read (list of strings).
    strReg : str
        String to search for, with regular expression; for instance
        "You have detected [0-9]+ out of [0-9]+ targets.".

    Returns
    -------
    varNumHit : int
        Number of successful trials.
    varNumMiss : int
        Number of unsuccessful trials.
    varRatio : float
        Ratio of successful to unsuccessful trials.
    """
    # Counter for hits:
    varCntHit = 0

    # Counter for trials:
    varCntTrial = 0

    # Loop through input files:
    for strIn in lstPath:

        # Open text file:
        fleIn = open(strIn, 'r')

        # Read file:
        csvIn = csv.reader(fleIn,
                           delimiter='\n',
                           skipinitialspace=True)

        # Create empty list for test:
        lstTxt = []
    
        # Loop through csv object to fill list:
        for lstTmp in csvIn:
            for strTmp in lstTmp:
                lstTxt.append(strTmp[:])

        # Close file:
        fleIn.close()

        # Filter out those lines that match the regular expression:
        lstTxt = [strTmp for strTmp in lstTxt if re.match(strReg, strTmp)]

        # Loop through lines:
        for strTmp01 in lstTxt:
            varHitTmp, varTrialTmp = [int(strTmp02) for strTmp02 in
                                      strTmp01.split() if strTmp02.isdigit()]

            # Update number of hits:
            varCntHit += varHitTmp
            
            # Update number of trials:
            varCntTrial += varTrialTmp

    # Number of hits:
    varNumHit = varCntHit

    # Number of misses:
    varNumMiss = varCntTrial - varCntHit

    # Success rate:
    varRatio = float(varNumHit) / float(varCntTrial)

    return varNumHit, varNumMiss, varRatio


