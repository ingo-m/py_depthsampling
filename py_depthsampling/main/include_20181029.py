# -*- coding: utf-8 -*-

"""
Adjust depth profiles from pilot session.

The depth profiles from pilot session 20181029 differ from following sessions.
The pilot session did not include the 'Kanizsa rotated' control condition.
Here, the data from the pilot session are integrated into the npz file
containing data from subsequent sessions. The npz file for the pilot session
first needs to be created separately from that of the following sessions.
"""

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


# -----------------------------------------------------------------------------
# ### Parameters

# Paths of npz file with depth profiles for all subjects except
# pilot session. (Meta-conditoin, ROI, condition left open.)
strPthNpz01 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/{}_{}.npz'  #noqa

# Path of npz file with depth profiles of pilot session (Meta-conditoin, ROI, condition left open.)
strPthNpz02 = '/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/{}/{}_{}_20181029.npz'  #noqa

# Meta-condition (within or outside of retinotopic stimulus area):
lstMtaCon = ['centre', 'edge', 'diamond', 'background']

# Region of interest ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

lstCon = ['bright_square_sst_pe',
          'kanizsa_rotated_sst_pe',
          'kanizsa_sst_pe']
lstCon = ['kanizsa_rotated_sst_pe']

# Array index of matching session:
varSesId01 = 2

# Session ID of pilot session:
strSesId02 = '20181029'
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# ### Loop through meta-conditions, ROIs, conditions

for idxMtaCon in lstMtaCon:
    for idxRoi in lstRoi:
        for idxCon in lstCon:

            # -----------------------------------------------------------------
            # ### Load data
            
            # Input path - npz file all sessions (except pilot session):
            strTmp01 = strPthNpz01.format(idxMtaCon, idxRoi, idxCon)

            # Input path - npz file pilot session:
            strTmp02 = strPthNpz02.format(idxMtaCon, idxRoi, idxCon)
        
            # Control condition 'Kanizsa rotated' was not included in the pilot
            # session. Fill the array with the corresponding data from the
            # same subject from a separate session (20181108).
                
            # Load depth profiles from disk - all sessions (except pilot
            # session):
            objNpz01 = np.load(strTmp01)
            aryDpth01 = objNpz01['arySubDpthMns']
            vecNumInc01 = objNpz01['vecNumInc']

            # Do not attempt to load 'Kanizsa rotated' condition for pilot
            # session:
            if not('kanizsa_rotated' in idxCon):

                # Load depth profiles from disk - pilot session:
                objNpz02 = np.load(strTmp02)
                aryDpth02 = objNpz02['arySubDpthMns']
                vecNumInc02 = objNpz02['vecNumInc']
            # -----------------------------------------------------------------

            # -----------------------------------------------------------------
            # ### Make adjustments

            # Append data from pilot session:
            if not('kanizsa_rotated' in idxCon):

                # Combine depth profiles:
                aryDpth = np.concatenate((aryDpth01, aryDpth02), axis=0)

                # Combine vectors with vertices included in profile:
                vecNumInc = np.concatenate((vecNumInc01, vecNumInc02), axis=0)

            # Append empty dummy array (in order to match array dimensions
            # across conditions):
            else:

                # Number of depth levels:
                varNumDpth = aryDpth01.shape[1]

                # Concatenate depth profiles:                
                aryDpth = np.concatenate(
                    (aryDpth01, aryDpth01[varSesId01, :].reshape(1,
                    varNumDpth)), axis=0)

                # Concatenate number of vertices:
                vecNumInc = np.concatenate(
                    (vecNumInc01, vecNumInc01[varSesId01].reshape(1)), axis=0)
            # -----------------------------------------------------------------               
                
            # -----------------------------------------------------------------
            # ### Save to disk

            np.savez(strTmp01,
                     arySubDpthMns=aryDpth,
                     vecNumInc=vecNumInc)
            # -----------------------------------------------------------------




