# -*- coding: utf-8 -*-
"""Project parameter estimates into a visual space representation."""

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


import os
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from py_depthsampling.project.load_par import load_par
from py_depthsampling.psf.project_ecc_par import project_ecc_par
from py_depthsampling.plot.plt_psf import plt_psf
from py_depthsampling.psf.fit_model import fitGauss
from py_depthsampling.psf.fit_model import fitLin
from py_depthsampling.psf.fit_model import funcGauss
from py_depthsampling.psf.fit_model import funcLin


# -----------------------------------------------------------------------------
# *** Define parameters

# Load/save existing projection from/to (ROI, condition, depth level label left
# open):
strPthNpy = '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf/{}_{}_{}.npz'  #noqa

# List of subject identifiers:
lstSubIds = ['20171023',  # '20171109',
             '20171204_01',
             '20171204_02',
             '20171211',
             '20171213',
             '20180111',
             '20180118']

# Nested list with depth levels to average over. For instance, if `lstDpth =
# [[0, 1, 2], [3, 4, 5]]`, on a first iteration, the average over the first
# three depth levels is calculated, and on a second iteration the average over
# the subsequent three depth levels is calculated. If 1lstDpth= [[None]]1,
# average over all depth levels.
lstDpth = [None, [0, 1, 2], [4, 5, 6], [8, 9, 10],
           [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
           [8, 8], [9, 9], [10, 10]]
lstDpth = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
           [8, 8], [9, 9], [10, 10]]

# Depth level condition labels (output file will contain this label):
lstDpthLbl = ['allGM', 'deepGM', 'midGM', 'superficialGM',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
lstDpthLbl = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# ROI ('v1' or 'v2'):
lstRoi = ['v1', 'v2', 'v3']

# Output path & prefix for plots (ROI, condition, depth level label left open):
strPthPltOt = '/Users/john/Dropbox/PacMan_Plots/project/psf_pe/{}_{}_{}'  #noqa

# File type suffix for plot:
# strFlTp = '.svg'
strFlTp = '.png'

# Figure scaling factor:
varDpi = 80.0

# Condition levels (used to complete file names):
# lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst',
#           'Pd_trn', 'Cd_trn', 'Ps_trn',
#           'Pd_min_Ps_sst', 'Pd_min_Cd_sst', 'Cd_min_Ps_sst',
#           'Pd_min_Cd_Ps_sst',
#           'Pd_min_Cd_Ps_trn',
#           'Pd_min_Ps_trn', 'Pd_min_Cd_trn', 'Cd_min_Ps_trn']
# lstCon = ['polar_angle', 'x_pos', 'y_pos', 'SD', 'R2']
lstCon = ['Pd_sst', 'Cd_sst', 'Ps_sst']

# Path of vtk mesh with data to project into visual space (e.g. parameter
# estimates; subject ID, hemisphere, and contion level left open).
strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/feat_level_2_{}_cope.vtk'  #noqa
# strPthData = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_{}.vtk'  #noqa

# Path of mean EPI (for scaling to percent signal change; subject ID and
# hemisphere left open):
strPthMneEpi = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/combined_mean.vtk'  #noqa

# Path of vtk mesh with R2 values from pRF mapping (at multiple depth levels;
# subject ID and hemisphere left open).
strPthR2 = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_R2.vtk'  #noqa

# Path of vtk mesh with pRF sizes (at multiple depth levels; subject ID and
# hemisphere left open).
strPthSd = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_SD.vtk'  #noqa

# Path of vtk mesh with pRF x positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthX = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_x_pos.vtk'  #noqa

# Path of vtk mesh with pRF y positions at multiple depth levels; subject ID
# and hemisphere left open).
strPthY = '/media/sf_D_DRIVE/MRI_Data_PhD/05_PacMan/{}/cbs/{}/pRF_results_y_pos.vtk'  #noqa

# Path of csv file with ROI definition (subject ID, hemisphere, and ROI left
# open).
strCsvRoi = '/home/john/PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa
# strCsvRoi = '/Users/john/1_PhD/GitLab/PacMan/analysis/{}/08_depthsampling/{}/{}_mod.csv'  #noqa

# Number of cortical depths.
varNumDpth = 11

# Extent of visual space from centre of the screen in negative x-direction
# (i.e. from the fixation point to the left end of the screen) in degrees of
# visual angle.
varExtXmin = -5.19
# Extent of visual space from centre of the screen in positive x-direction
# (i.e. from the fixation point to the right end of the screen) in degrees of
# visual angle.
varExtXmax = 5.19
# Extent of visual space from centre of the screen in negative y-direction
# (i.e. from the fixation point to the lower end of the screen) in degrees of
# visual angle.
varExtYmin = -5.19
# Extent of visual space from centre of the screen in positive y-direction
# (i.e. from the fixation point to the upper end of the screen) in degrees of
# visual angle.
varExtYmax = 5.19

# R2 threshold for vertex inclusion (vertices with R2 value below threshold are
# not considered for plot):
varThrR2 = 0.15

# Number of eccentricity bins for visual space representation:
varNumEcc = 1000

# Plot parameters over this eccentricity range:
tplRngEcc = (0.0, 6.0) #(2.0, 5.5)

# Normalise data for plots? (In order to make compairson of width/shape of
# profiles more easy, plots can be scaled to a common range between zero and
# one.)
lgcNorm = False

# Fit function (None, 'gaussian', or 'linear')?
strFit = 'linear'

# Restrict model fitting to this range (to use same range as `tplRngEcc`, set
# `tplFitRng = None`).
tplFitRng = (2.2, 3.75)

# Save result from model fitting (i.e. slope or width of function) to disk
# (pandas data frame saved as JSON for import in R). If `None`, data frame is
# not created.
strPthJson = '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf/dataframe.json'  #noqa
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Parent loop

print('-Plot parameter estimates by eccentricity')

# Number of subjects/ROIs/conditions:
varNumSub = len(lstSubIds)
varNumRoi = len(lstRoi)
varNumCon = len(lstCon)

# Create dataframe & save JSON?
if (not (strPthJson is None)) and (not (strFit is None)):

    print('--Creatdataframe for model width/slope.')

    # Total number of samples for dataframe:
    varNumSmpl = (varNumDpth * varNumRoi * varNumCon)

    # Feature list (column names for dataframe):
    lstFtr = ['ROI', 'Condition', 'Depth', 'Slope']

    # Dataframe:
    objDf = pd.DataFrame(0.0, index=np.arange(varNumSmpl), columns=lstFtr)

    # Coutner:
    idxSmpl = 0

# Loop through depth levels, ROIs, and conditions:
for idxDpth in range(len(lstDpth)):  #noqa
    for idxRoi in range(len(lstRoi)):
        for idxCon in range(len(lstCon)):

            # File name of npy file for current condition:
            strPthNpyTmp = strPthNpy.format(lstRoi[idxRoi],
                                            lstCon[idxCon],
                                            lstDpthLbl[idxDpth])

            if os.path.isfile(strPthNpyTmp):

                print('--Load existing visual field projection')

                # Load existing projection:
                objNpz = np.load(strPthNpyTmp)
                vecVslSpc = objNpz['vecVslSpc']
                vecNorm = objNpz['vecNorm']

            else:

                # -------------------------------------------------------------
                # *** Load data

                print('--Load data from vtk meshes')

                # Number of processes to run in parallel:
                varPar = varNumSub

                # Create a queue to put the results in:
                queOut = mp.Queue()

                # Empty list for processes:
                lstPrcs = [None] * varPar

                # Empty list for results of parallel processes:
                lstRes = [None] * varPar

                # Create processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc] = mp.Process(target=load_par,
                                                 args=(lstSubIds[idxPrc],
                                                       lstCon[idxCon],
                                                       lstRoi[idxRoi],
                                                       strPthData,
                                                       strPthMneEpi,
                                                       strPthR2,
                                                       strPthX,
                                                       strPthY,
                                                       strPthSd,
                                                       strCsvRoi,
                                                       varNumDpth,
                                                       lstDpth[idxDpth],
                                                       idxPrc,
                                                       queOut)
                                                 )

                    # Daemon (kills processes when exiting):
                    lstPrcs[idxPrc].Daemon = True

                # Don't create more processes than number of subjects:
                # varParTmp = int(np.min([varPar, len(lstSubIds)]))

                # Start processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc].start()

                # Collect results from queue:
                for idxPrc in range(varPar):
                    lstRes[idxPrc] = queOut.get(True)

                # Join processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc].join()

                # List for single-subject data vectors:
                lstData = [None] * varPar

                # List for single-subject mean EPI vectors:
                lstMneEpi = [None] * varPar

                # List of single subject R2 vectors:
                lstR2 = [None] * varPar

                # List for single subject SD vectors (pRF sizes):
                lstSd = [None] * varPar

                # List for single subject x-position vectors:
                lstX = [None] * varPar

                # List for single subject y-position vectors:
                lstY = [None] * varPar

                # Put output into correct order (unnecessary in this context
                # but kept for consistency):
                for idxRes in range(varPar):

                    # Index of results (first item in output list):
                    varTmpIdx = lstRes[idxRes][0]

                    # Put fitting results into list, in correct order:
                    lstData[varTmpIdx] = lstRes[idxRes][1]
                    lstMneEpi[varTmpIdx] = lstRes[idxRes][2]
                    lstR2[varTmpIdx] = lstRes[idxRes][3]
                    lstSd[varTmpIdx] = lstRes[idxRes][4]
                    lstX[varTmpIdx] = lstRes[idxRes][5]
                    lstY[varTmpIdx] = lstRes[idxRes][6]

                # Concatenate arrays from all subjects:
                vecData = np.concatenate(lstData[:])
                vecMneEpi = np.concatenate(lstMneEpi[:])
                vecR2 = np.concatenate(lstR2[:])
                vecSd = np.concatenate(lstSd[:])
                vecX = np.concatenate(lstX[:])
                vecY = np.concatenate(lstY[:])

                # Delete original lists:
                del(lstData)
                del(lstMneEpi)
                del(lstR2)
                del(lstSd)
                del(lstX)
                del(lstY)

                # -------------------------------------------------------------
                # *** Convert cope to percent signal change

                # According to the FSL documentation
                # (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide), the
                # PEs can be scaled to signal change with respect to the mean
                # (over time within voxel): "This is achieved by scaling the PE
                # or COPE values by (100*) the peak-peak height of the
                # regressor (or effective regressor in the case of COPEs) and
                # then by dividing by mean_func (the mean over time of
                # filtered_func_data)."

                # Only perform scaling if the data is from an FSL cope file:
                if 'cope' in strPthData:
                    print('--Convert cope to percent signal change.')

                    # The peak-peak height depends on the predictor (i.e.
                    # condition).
                    if 'sst' in lstCon[idxCon]:
                        varPpheight = 1.268049
                    elif 'trn' in lstCon[idxCon]:
                        varPpheight = 0.2269044

                    # In order to avoid division by zero, avoid zero-voxels:
                    lgcTmp = np.not_equal(vecData, 0.0)

                    # Apply PSC scaling, as described above:
                    vecData[lgcTmp] = \
                        np.multiply(
                                    np.divide(
                                              np.multiply(
                                                          vecData[lgcTmp],
                                                          (100.0 * varPpheight)
                                                          ),
                                              vecMneEpi[lgcTmp]),
                                    1.0  # 1.4
                                    )

                # -------------------------------------------------------------
                # *** Project data into visual space

                print('--Project data into visual space')

                # Number of processes to run in parallel:
                varPar = 11

                # Split data into chunks:
                lstData = np.array_split(vecData, varPar)
                lstR2 = np.array_split(vecR2, varPar)
                lstSd = np.array_split(vecSd, varPar)
                lstX = np.array_split(vecX, varPar)
                lstY = np.array_split(vecY, varPar)

                # Create a queue to put the results in:
                queOut = mp.Queue()

                # Empty list for processes:
                lstPrcs = [None] * varPar

                # Empty list for results of parallel processes:
                lstRes = [None] * varPar

                # Create processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc] = mp.Process(target=project_ecc_par,
                                                 args=(idxPrc,
                                                       lstData[idxPrc],
                                                       lstX[idxPrc],
                                                       lstY[idxPrc],
                                                       lstSd[idxPrc],
                                                       lstR2[idxPrc],
                                                       varThrR2,
                                                       varNumEcc,
                                                       varExtXmin,
                                                       varExtXmax,
                                                       varExtYmin,
                                                       varExtYmax,
                                                       queOut)
                                                 )

                    # Daemon (kills processes when exiting):
                    lstPrcs[idxPrc].Daemon = True

                # Start processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc].start()

                # Collect results from queue:
                for idxPrc in range(varPar):
                    lstRes[idxPrc] = queOut.get(True)

                # Join processes:
                for idxPrc in range(varPar):
                    lstPrcs[idxPrc].join()

                # List for results after re-ordering (visual space arrays and
                # normalisation arrays):
                lstVslSpc = [None] * varPar
                lstNorm = [None] * varPar

                # Put output into correct order (unnecessary in this context
                # but kept for consistency):
                for idxRes in range(varPar):

                    # Index of results (first item in output list):
                    varTmpIdx = lstRes[idxRes][0]

                    # Put fitting results into list, in correct order:
                    lstVslSpc[varTmpIdx] = lstRes[idxRes][1]
                    lstNorm[varTmpIdx] = lstRes[idxRes][2]

                # Visual space array (2D array with bins of locations in visual
                # space):
                vecVslSpc = np.zeros((varNumEcc))

                # Array for normalisation (parameter estimates are summed up
                # over the visual field; the normalisation array is needed to
                # normalise the sum):
                vecNorm = np.zeros((varNumEcc))

                # Add up results from separate processes:
                for idxPrc in range(varPar):
                    vecVslSpc = np.add(lstVslSpc[idxPrc], vecVslSpc)
                    vecNorm = np.add(lstNorm[idxPrc], vecNorm)

                # Normalise:
                vecVslSpc = np.divide(vecVslSpc, vecNorm)

                # Save results to disk:
                np.savez(strPthNpyTmp,
                         vecVslSpc=vecVslSpc,
                         vecNorm=vecNorm)

            # -----------------------------------------------------------------
            # *** Plot results

            print('--Plot results')

            # Maximum eccentricity in visual field:
            varEccMax = np.sqrt(
                                np.add(
                                       np.power(varExtXmax, 2.0),
                                       np.power(varExtYmax, 2.0)
                                       )
                                )

            # Vector with visual space coordinates of elements in `vecVslSpc`
            # (eccentricity in deg of visual angle):
            vecCorEcc = np.linspace(0.0,
                                    varEccMax,
                                    num=varNumEcc,
                                    endpoint=True)

            # Array indices of minimum & maximum values (with respect to
            # x-axis, i.e. eccentricity):
            varIdxMin = (np.abs(vecCorEcc - tplRngEcc[0])).argmin()
            varIdxMax = (np.abs(vecCorEcc - tplRngEcc[1])).argmin()

            # -----------------------------------------------------------------
            # * Normalise

            # Normalise plots (to common range on y-axis, from zero to one):
            if lgcNorm:

                # If function is fitted, use range over which function is
                # fitted for normalisation:
                if not (strFit is None):

                    # Array indices of minimum & maximum values (with respect
                    # to x-axis, i.e. eccentricity):
                    varIdxFitMin = (np.abs(vecCorEcc - tplFitRng[0])).argmin()
                    varIdxFitMax = (np.abs(vecCorEcc - tplFitRng[1])).argmin()

                    varMin = np.min(vecVslSpc[varIdxFitMin:varIdxFitMax])
                    vecVslSpc = np.subtract(vecVslSpc, varMin)
                    varMax = np.max(vecVslSpc[varIdxFitMin:varIdxFitMax])
                    vecVslSpc = np.divide(vecVslSpc, varMax)

                # Otherwise, use plot x-range for normalisation.
                else:

                    varMin = np.min(vecVslSpc[varIdxMin:varIdxMax])
                    vecVslSpc = np.subtract(vecVslSpc, varMin)
                    varMax = np.max(vecVslSpc[varIdxMin:varIdxMax])
                    vecVslSpc = np.divide(vecVslSpc, varMax)

            # -----------------------------------------------------------------
            # * Fit Gaussian

            # Fit any function?
            if strFit is None:

                # If no function is fitted, don't plot legend:
                lgcLgnd = False
                lstConLbl = None

            else:

                lgcLgnd = True

                # If no range for function fitting is provided, set to same
                # range as used for plot:
                if tplFitRng is None:
                    tplFitRng = tplRngEcc

                # Array indices of minimum & maximum values (with respect to
                # x-axis, i.e. eccentricity):
                varIdxFitMin = (np.abs(vecCorEcc - tplFitRng[0])).argmin()
                varIdxFitMax = (np.abs(vecCorEcc - tplFitRng[1])).argmin()

                # Predicted values (NaNs in same shape as visual space):
                vecMdl = np.full(vecCorEcc.shape, np.nan)

                if strFit == 'gaussian':

                    # Fit Gaussian function to relevant section of visual
                    # space:
                    varMu, varSd, varInt = \
                        fitGauss(vecCorEcc[varIdxFitMin:varIdxFitMax],
                                 vecVslSpc[varIdxFitMin:varIdxFitMax])

                    # Only produce predicted values for specified range:
                    vecMdl[varIdxFitMin:varIdxFitMax] = \
                        funcGauss(vecCorEcc[varIdxFitMin:varIdxFitMax], varMu,
                                  varSd, varInt)

                    # Use list of condition labels to show model parameters.
                    lstConLbl = [('Model, SD = ' + str(np.around(varSd,
                                  decimals=2))),
                                 'Empirical']

                elif strFit == 'linear':

                    # Fit linear function to relevant section of visual space:
                    varSlp, varInt = \
                        fitLin(vecCorEcc[varIdxFitMin:varIdxFitMax],
                               vecVslSpc[varIdxFitMin:varIdxFitMax])

                    # Only produce predicted values for specified range:
                    vecMdl[varIdxFitMin:varIdxFitMax] = \
                        funcLin(vecCorEcc[varIdxFitMin:varIdxFitMax], varSlp,
                                varInt)

                    # Use list of condition labels to show model parameters.
                    lstConLbl = [('Model, slope = ' + str(np.around(varSlp,
                                  decimals=2))),
                                 'Empirical']

                # Bring predicted values into shape of visual space to be
                # plotted:
                vecMdl = vecMdl[varIdxMin:varIdxMax]

                # We only save slopes for single depth levels (i.e. not for
                # 'allGM', 'deepGM', etc.). List with labels used for single
                # depth levels for check:
                lstSnglDpthLbl = [str(x) for x in range(varNumDpth)]

                # Create dataframe & save JSON?
                lgcTmp = ((not (strPthJson is None))
                          and (lstDpthLbl[idxDpth] in lstSnglDpthLbl))

                if lgcTmp:

                    # Features to dataframe:
                    objDf.at[idxSmpl, 'ROI'] = idxRoi
                    objDf.at[idxSmpl, 'Condition'] = idxCon
                    objDf.at[idxSmpl, 'Depth'] = lstDpth[idxDpth][0]

                    # Width or slope to dataframe:
                    if strFit == 'gaussian':
                        objDf.at[idxSmpl, 'Slope'] = varSd
                    elif strFit == 'linear':
                        objDf.at[idxSmpl, 'Slope'] = varSlp

                    idxSmpl += 1

            # Section of visual space to be plotted:
            vecVslSpcTmp = np.array(vecVslSpc[varIdxMin:varIdxMax], ndmin=2)

            # Section of normalisation vector to be plotted:
            vecNormTmp = vecNorm[varIdxMin:varIdxMax]
            # Maximum normalisation value across range of visual space:
            varNormMax = np.max(vecNormTmp)
            # Scale to maximum of one:
            vecNormTmp = np.divide(vecNormTmp, varNormMax)
            # Invert:
            vecNormTmp = np.subtract(1.0, vecNormTmp)
            # Array shape:
            vecNormTmp = np.array(vecNormTmp, ndmin=2)

            if strFit:

                # Add predicted values to array to be plotted:
                vecVslSpcTmp = np.stack((vecMdl,
                                         vecVslSpcTmp.flatten()))

                # Add dummy vector to normalisation vector:
                vecNormTmp = np.stack((np.zeros((vecNormTmp.shape[1])),
                                       vecNormTmp.flatten()))

            # x-axis values for section of visual space to be plotted:
            vecCorEccTmp = vecCorEcc[varIdxMin:varIdxMax]

            # Output path for plot:
            strPthPltOtTmp = (strPthPltOt.format(lstRoi[idxRoi],
                                                 lstCon[idxCon],
                                                 lstDpthLbl[idxDpth])
                              + strFlTp)

            # Plot title:
            strTmpTtl = (lstRoi[idxRoi]
                         + ' '
                         + lstCon[idxCon]
                         + ' '
                         + lstDpthLbl[idxDpth])

            # Create plot:
            plt_psf(vecVslSpcTmp,
                    strPthPltOtTmp,
                    varYmin=-5.0,
                    varYmax=1.0,
                    vecX=vecCorEccTmp,
                    aryError=vecNormTmp,
                    lgcLgnd=lgcLgnd,
                    lstConLbl=lstConLbl,
                    strXlabel='Eccentricity',
                    strYlabel='Signal change [%]',
                    strTitle=strTmpTtl,
                    varNumLblY=5,
                    varPadY=(0.1, 0.1),
                    lstVrt=[3.75])

# Create dataframe & save JSON?
if (not (strPthJson is None)) and (not (strFit is None)):

    print('--Saving dataframe to json.')

    # Save dataframe to json:
    objDf.to_json(strPthJson)

    # Output path & prefix for plots (ROI, condition, depth level label left
    # open):
    strPthTmp = (strPthPltOt.format(strFit, 'model', 'fit_by_ROI') + strFlTp)

    # Create seaborn colour palette:
    colors = ["amber", "greyish", "faded green"]
    objClr = sns.xkcd_palette(colors)

    # Draw nested barplot:
    fgr01 = sns.factorplot(x="ROI", y="Slope", hue="Condition", data=objDf,
                           size=6, kind="bar", palette=objClr)

    # Set x-axis labels to upper case ROI labels:
    lstRoiUp = [x.upper() for x in lstRoi]
    fgr01.set_xticklabels(lstRoiUp)

    # Set hue labels (i.e. condition labels in legend):
    for objTxt, strLbl in zip(fgr01._legend.texts, lstCon):
        objTxt.set_text(strLbl)

    # Save figure:
    fgr01.savefig(strPthTmp)

    # Output path & prefix for plots (ROI, condition, depth level label left
    # open):
    strPthTmp = (strPthPltOt.format(strFit, 'model', 'fit_by_Depth') + strFlTp)

    # Create seaborn colour palette:
    objClr = sns.light_palette((210, 90, 60), input="husl",
                               n_colors=varNumDpth)

    # Draw nested barplot:
    fgr02 = sns.factorplot(x="ROI", y="Slope", hue="Depth", data=objDf, size=6,
                           kind="bar", legend=True, palette=objClr)

    fgr02.set_xticklabels(lstRoiUp)

    # Save figure:
    fgr02.savefig(strPthTmp)
# -----------------------------------------------------------------------------
