# -*- coding: utf-8 -*-
"""Utilities for parallelised loading of vtk files."""

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


# ----------------------------------------------------------------------------
# *** Import modules

import numpy as np
from ds_loadVtkSingle import funcLoadVtkSingle
from ds_loadVtkMulti import funcLoadVtkMulti


# ----------------------------------------------------------------------------
# *** Functions

def load_multi_vtk_par(idxPrc, strVtkIn, varNumDpth, strPrcdData, varNumLne,
                       queOut):
    """
    Wrapper function to load multi-depth-level vtk files in parallel.
    
    Parameters
    ----------
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
    strVtkIn : str
        Path of vtk file to load (vtk file with multiple values per vertex).
    varNumDpth : int
        Number of data points per vertex (i.e. number of depth levels).
    strPrcdData : str
        Beginning of string which precedes vertex data in data vtk files (i.e.
        in the statistical maps).
    varNumLne : int
        Number of lines between vertex-identification-string and first data
        point.
    queOut : multiprocessing.queues.Queue
        Queue to put results on.

    Returns
    -------
    lstOut : list
        List with results, containing the following objects:
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
    aryDpth : np.array
        Array with depth-sampled data, of the form aryDpth[idxVrtx, idxDpth].
    """
    # Load data from disk:
    aryDpth = funcLoadVtkMulti(strVtkIn, strPrcdData, varNumLne, varNumDpth)

    # Output list:
    lstOut = [idxPrc, aryDpth]

    # Put output on list:
    queOut.put(lstOut)



def load_single_vtk_par(idxPrc, strVtkIn, strPrcdData, varNumLne, queOut):
    """
    Wrapper function to load single-depth-level vtk files in parallel.
    
    Parameters
    ----------
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
    strVtkIn : str
        Path of vtk file to load (vtk file with one value per vertex).
    strPrcdData : str
        Beginning of string which precedes vertex data in data vtk files (i.e.
        in the statistical maps).
    varNumLne : int
        Number of lines between vertex-identification-string and first data
        point.
    queOut : multiprocessing.queues.Queue
        Queue to put results on.

    Returns
    -------
    lstOut : list
        List with results, containing the following objects:
    idxPrc : int
        Process ID of parallel process. Needed to put output in order in
        parent function.
    aryVtk : np.array
        Array with vtk data, of the form aryVtk[idxVrtx, 1].
    """
    # Load data from disk:
    aryVtk = funcLoadVtkSingle(strVtkIn, strPrcdData, varNumLne)

    # Output list:
    lstOut = [idxPrc, aryVtk]

    # Put output on list:
    queOut.put(lstOut)
