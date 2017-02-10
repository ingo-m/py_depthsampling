# -*- coding: utf-8 -*-

"""
VTK depth samling across subjects.

The purpose of this script is to visualise cortical depth sampling results from
vtk files. The script can visualise statistical maps from vtk files. Vertices
are selected according to several criteria. Two vertex selection criteria are
always applied:

    (1) The vertex has to be contained within the ROI (as defined by by a csv
        file).
    (2) The vertex has to surpass some intensity criterion (e.g. retinotopic
        overlap above a certain level).

Other optional vertex selection criteria are:

    (3) Multi-depth level criterion I -  vertices that are BELOW a certain
        threshold at any depth levels can be excluded. For example, a venogram,
        (or a T2* weighted EPI image with low intensities around veins) that is
        defined at all depth levels can be used. If the vertex values are below
        a threshold at any depth level, the vertex is excluded.
    (4) Multi-depth level criterion II - vertices that are ABOVE a certain
        threshold across depth levels can be excluded. For instance, a mask
        that is defined at all depth levels can be used. If the vertex value
        is above a threshold at any depth level, the vertex is excluded.

    (5) Multi-level data distribution criterion I
        Selection based on combination of z-conjunction-mask mask and
        distribution of z-values.
    (6) Multi-level data distribution criterion II
        Calculates maximum data value across depth levels, and excludes
        vertices whose across-depth-maximum-value is at the lower and/or upper
        end of the distribution across vertices (as specified by the user).
        (The distribution across those vertices that have survived all previous
        exclusion criteria is used here.)
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


# *****************************************************************************
# *** Import modules

import numpy as np
import multiprocessing as mp
from ds_acrSubsGetData import funcAcrSubGetSubsData
from ds_pltAcrSubsMean import funcPltAcrSubsMean
from ds_pltAcrSubsLinReg import funcPltAcrSubsLinReg
# *****************************************************************************

print('-Visualisation of depth sampling results')

# *****************************************************************************
# *** Define parameters

# List of subject identifiers:
lstSubIds = ['20150930',
             '20151118',
             '20151127_01',
             '20151130_01',
             '20151130_02',
             '20161205',
             '20161207',
             '20161212_01',
             '20161212_02',
             '20161214',
             '20161219_01']

# List of first set of vtk files with depth-sampled data (list of lists), e.g.
# parameter estimates:
lstVtkDpth01 = [
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                  '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                  '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                  '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_pe_stim_lvl_04.vtk'],  #noqa
                ]

# (1)
# Path of csv files with ROI definition (i.e. patch of cortex selected on the
# surface, e.g. V1) - i.e. the first vertex selection criterion:
lstCsvRoi = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/v1.csv',  #noqa
             '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/v1.csv']  #noqa

# (2)
# Use second selection criterion (e.g. retinotopic information)?
lgcSlct02 = True
# Path of vtk files with 2nd vertex selection criterion (e.g. retinotopic
# information, such as vertices' pRF overlap with visual stimulus). This vtk
# file is supposed to contain one set of data values (e.g. retinotopic
# information at mid-grey-matter).
lstVtkSlct02 = ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_R2.vtk',  #noqa
                '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_R2.vtk']  #noqa
# Threshold for vertex selection for 2nd selection criterion (e.g. minimum pRF
# overlap):
varThrSlct02 = 0.2

# (3)
# Use third selection criterion (e.g. z-conjunction-mask)?
lgcSlct03 = False
# Path of vtk files with 3rd vertex selection criterion (e.g. z-conjunction
# mask). This vtk file is supposed to contain one set of data values for each
# depth level (e.g. value of z-conjunction mask at each depth level).
lstVtkSlct03 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']  #noqa
# Threshold for vertex selection for second selection criterion:
varThrSlct03 = 0.5

# (4)
# Use exclusion vtk mask?
lgcMskExcl = False
# Path of exclusion mask:
lstVtkExcl = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']  #noqa
# Exclusion mask threshold (if ABOVE this threshold at any depth level, vertex
# is excluded from depth sampling):
varThrExcl = 0.5

# (5)
# Load second set of vtk data files (z-values) and use them for vertex
# selection?
lgcVtk02 = True
# How many vertices to select for each subject?
lstNumVrtx = [1000] * 11
# List of second set of vtk files with depth-sampled data (list of lists), e.g.
# z-values:
lstVtkDpth02 = [
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20150930/cbs_distcor/lh/20150930_mp2rage_seg_v24_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151118/cbs_distcor/lh/20151118_mp2rage_seg_v14_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151127_01/cbs_distcor/lh/20151127_01_mp2rage_seg_v15_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_01/cbs_distcor/lh/20151130_01_mp2rage_seg_v16_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20151130_02/cbs_distcor/lh/20151130_02_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161205/cbs_distcor/lh/20161205_mp2rage_seg_v20_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161207/cbs_distcor/lh/20161207_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_01/cbs_distcor/lh/20161212_01_mp2rage_seg_v32_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161212_02/cbs_distcor/lh/20161212_02_mp2rage_seg_v28_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161214/cbs_distcor/lh/20161214_mp2rage_seg_v31_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ['/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_01.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_02.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_03.vtk',  #noqa
                 '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/20161219_01/cbs_distcor/lh/20161219_01_mp2rage_seg_v19_lh__surf_05_inf_zstat_lvl_04.vtk'],  #noqa
                ]

# (6)
# Use PE range?
lgcPeRng = False
# Lower bound of PE range (vertices with a maximum PE across depths that
# is below this percentile in the distribution of those values across
# vertices will be excluded):
varPeRngLw = 0.0
# Upper bound of PE range (vertices with a maximum PE across depths that
# is above this percentile in the distribution of those values across
# vertices will be excluded):
varPeRngUp = 90.0

# Number of header lines in ROI CSV file:
varNumHdrRoi = 1

# Number of cortical depths:
varNumDpth = 11

# Beginning of string which precedes vertex data in data vtk files (i.e. in the
# statistical maps):
strPrcdData = 'SCALARS'

# Number of lines between vertex-identification-string and first data point:
varNumLne = 2

# Title for mean plot:
strTitle = ('Left hemisphere V1')

# Limits of y-axis for single subject plots (list of tuples, [(Ymin, Ymax)]):
lstLimY = [(0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 3.0),
           (0.0, 2.5),
           (0.0, 2.5),
           (0.0, 2.5)]

# Limits of y-axis for across subject plot:
varAcrSubsYmin = -0.05
varAcrSubsYmax = 2.0  # 1.90

# Label for axes:
strXlabel = 'Cortical depth level (equivolume)'
strYlabel = 'fMRI signal change [arbitrary units]'

# Condition labels:
lstConLbl = ['2.5%', '6.1%', '16.3%', '72.0%']

# Output path for plots - prfix:
strPltOtPre = '/home/john/Desktop/tex_acrs_subs_zconj/plots_v1/'
# Output path for plots - suffix:
strPltOtSuf = '_depthplot_R2_0p2_z_con_2000_vertices.png'

# Figure scaling factor:
varDpi = 96.0

# If normalisation - data from which input file to divide by?
# (Indexing starts at zero.) Note: This functionality is not used at the
# moment. Instead of dividing by a reference condition, all profiles are
# divided by the grand mean within subjects before averaging across subjects
# (if lgcNormDiv is true).
varNormIdx = 0

# Normalise by division?
lgcNormDiv = True

# Linear regression?
lgcLinReg = False

# Constrast vector for simple linear regression model (array with one value per
# condition, e.g. per stimulus contrast level):
vecLinRegMdl = np.array([-3.0, -1.0, 1.0, 3.0])

# P-threshold (uncorrected) for regression model (will be corrected for
# multiple comparison automatically): *** NOT IMPLEMENTED
varLinRegP = 0.001

# Range of y-axis for regression plots:
varLinRegYmin = 0.025  # 0.085  # 0.025  # 0.09
varLinRegYmax = 0.165  # 0.165  # 0.15  # 0.17

# Linear regression plot - label for axes:
strLinRegYlabel = 'Regression coefficient'

# Output path for depth samling results (within subject means):
strDpthMeans = '/media/sf_D_DRIVE/MRI_Data_PhD/04_ParCon/Higher_Level_Analysis/depthsampling_mean_pe/v1.npy'  #noqa

# Maximum number of processes to run in parallel: *** NOT IMPLEMENTED
# varPar = 10
# *****************************************************************************


# *****************************************************************************
# *** Plot and retrieve single subject data

print('---Plotting & retrieving single subject data')

# Number of subjects:
varNumSubs = len(lstSubIds)

# List for single subject data (mean PE over depth levels):
lstSubData01 = [None] * varNumSubs

# Empty list to collect results from parallelised function:
lstParResult = [None] * varNumSubs

# Empty list for processes:
lstPrcs = [None] * varNumSubs

# Create a queue to put the results in:
queOut = mp.Queue()

# Loop through subjects:
for idxPrc in range(0, varNumSubs):

    # Prepare processes that plot & return single subject data:
    lstPrcs[idxPrc] = \
        mp.Process(target=funcAcrSubGetSubsData,
                   args=(idxPrc,                # Process ID
                         lstSubIds[idxPrc],     # Data struc - Subject ID
                         lstVtkDpth01[idxPrc],  # Data struc - Pth vtk I
                         lstCsvRoi[idxPrc],     # Data struc - ROI CSV fle
                         varNumDpth,    # Data struc - Num. depth levels
                         varNumHdrRoi,  # Data struc - Header lines CSV
                         strPrcdData,   # Data struc - Str. prcd. VTK data
                         varNumLne,     # Data struc - Lns. prcd. data VTK
                         lgcSlct02,             # Criterion 2 - Yes or no?
                         lstVtkSlct02[idxPrc],  # Criterion 2 - VTK path
                         varThrSlct02,          # Criterion 2 - Threshold
                         lgcSlct03,             # Criterion 3 - Yes or no?
                         lstVtkSlct03[idxPrc],  # Criterion 3 - VTK path
                         varThrSlct03,          # Criterion 3 - Threshold
                         lgcMskExcl,            # Criterion 4 - Yes or no?
                         lstVtkExcl[idxPrc],    # Criterion 4 - VTK path
                         varThrExcl,            # Criterion 4 - Threshold
                         lgcVtk02,              # Criterion 5 - Yes or no?
                         lstVtkDpth02[idxPrc],  # Criterion 5 - VTK path
                         lstNumVrtx[idxPrc],    # Criterion 5 - Num vrtx
                         lgcPeRng,              # Criterion 6 - Yes or no?
                         varPeRngLw,            # Criterion 6 - Lower lim
                         varPeRngUp,            # Criterion 6 - Upper lim
                         lgcNormDiv,    # Normalisation - Yes or no?
                         varNormIdx,    # Normalisation - Which reference
                         varDpi,              # Plot - dots per inch
                         lstLimY[idxPrc][0],  # Plot - Minimum of Y axis
                         lstLimY[idxPrc][1],  # Plot - Maximum of Y axis
                         lstConLbl,           # Plot - Condition labels
                         strXlabel,           # Plot - X axis label
                         strYlabel,           # Plot - Y axis label
                         strTitle,            # Plot - Title
                         strPltOtPre,   # Plot - Output file path prefix
                         strPltOtSuf,   # Plot - Output file path suffix
                         queOut)        # Queue for output list
                   )

    # Daemon (kills processes when exiting):
    lstPrcs[idxPrc].Daemon = True

# Start processes:
for idxPrc in range(0, varNumSubs):
    lstPrcs[idxPrc].start()

# Collect results from queue:
for idxPrc in range(0, varNumSubs):
    lstParResult[idxPrc] = queOut.get(True)

# Join processes:
for idxPrc in range(0, varNumSubs):
    lstPrcs[idxPrc].join()

# Create list  to put the function output into the correct order:
lstPrcId = [None] * varNumSubs
lstSubData01 = [None] * varNumSubs

# Put output into correct order:
for idxRes in range(0, varNumSubs):

    # Index of results (first item in output list):
    varTmpIdx = lstParResult[idxRes][0]

    # Put fitting results into list, in correct order:
    lstSubData01[varTmpIdx] = lstParResult[idxRes][1]

# Number of conditions (i.e. number of data vtk files per subject):
varNumCon = len(lstVtkDpth01[0])

# Array with single-subject depth sampling results, of the form
# aryDpthMeans[idxSub, idxCondition, idxDpth].
arySubDpthMns = np.zeros((varNumSubs, varNumCon, varNumDpth))

# Retrieve single-subject data from list:
for idxSub in range(0, varNumSubs):
            arySubDpthMns[idxSub, :, :] = lstSubData01[idxSub]
# *****************************************************************************


# *****************************************************************************
# *** Save results

# We save the mean parameter estimates of each subject to disk. This file can
# be used to plot results from different ROIs in one plot.

np.save(strDpthMeans, arySubDpthMns)
# *****************************************************************************


# *****************************************************************************
# *** Plot mean over subjects

print('---Plot results - mean over subjects.')

funcPltAcrSubsMean(arySubDpthMns,
                   varNumSubs,
                   varNumDpth,
                   varNumCon,
                   varDpi,
                   varAcrSubsYmin,
                   varAcrSubsYmax,
                   lstConLbl,
                   strXlabel,
                   strYlabel,
                   strTitle,
                   strPltOtPre,
                   strPltOtSuf)
# *****************************************************************************


# *****************************************************************************
# *** Plot mean simple regression (across-subjects)

if lgcLinReg:

    print('---Plot results - simple linear regression (across-subjects).')

    funcPltAcrSubsLinReg(arySubDpthMns,
                         vecLinRegMdl,
                         varNumSubs,
                         varNumDpth,
                         strTitle,
                         strXlabel,
                         strLinRegYlabel,
                         varLinRegYmin,
                         varLinRegYmax,
                         varLinRegP,
                         varDpi,
                         strPltOtPre,
                         strPltOtSuf)
# *****************************************************************************
