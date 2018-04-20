"""Function of the depth sampling pipeline."""

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
from scipy.interpolate import griddata
from py_depthsampling.plot.plt_acr_dpth import plt_acr_dpth
from py_depthsampling.plot.plt_dpth_prfl_acr_subs import plt_dpth_prfl_acr_subs
from py_depthsampling.drain_model.drain_model_decon_01 import deconv_01
from py_depthsampling.drain_model.drain_model_decon_02 import deconv_02
from py_depthsampling.drain_model.drain_model_decon_03 import deconv_03
from py_depthsampling.drain_model.drain_model_decon_04 import deconv_04
from py_depthsampling.drain_model.drain_model_decon_05 import deconv_05
from py_depthsampling.drain_model.drain_model_decon_06 import deconv_06
from py_depthsampling.drain_model.find_peak import find_peak


def drain_model(varMdl, strRoi, strHmsph, strPthPrf, strPthPrfOt, strPthPltOt,  #noqa
                strFlTp, varDpi, strXlabel, strYlabel, lstCon, lstConLbl,
                varNumIt, varCnfLw, varCnfUp, varNseRndSd, varNseSys, lstFctr,
                varAcrSubsYmin01, varAcrSubsYmax01, varAcrSubsYmin02,
                varAcrSubsYmax02):
    """Model-based correction of draining effect."""
    # ----------------------------------------------------------------------------
    # *** Load depth profile from disk

    print('-Model-based correction of draining effect')

    print('---Loading data')

    # Array with single-subject depth sampling results, of the form
    # aryEmpSnSb[idxSub, idxCondition, idxDpth].
    aryEmpSnSb = np.load(strPthPrf)

    # Number of subjects:
    varNumSub = aryEmpSnSb.shape[0]

    # Number of conditions:
    varNumCon = aryEmpSnSb.shape[1]

    # Number of equi-volume depth levels in the input data:
    varNumDpth = aryEmpSnSb.shape[2]

    # -------------------------------------------------------------------------
    # *** Subject-by-subject deconvolution

    print('---Subject-by-subject deconvolution')

    # Array for single-subject interpolation result (before deconvolution):
    aryEmp5SnSb = np.zeros((varNumSub, varNumCon, 5))

    if (varMdl != 4) and (varMdl != 5) and (varMdl != 6):
        # Array for single-subject deconvolution result (defined at 5 depth
        # levels):
        aryDecon5 = np.zeros((varNumSub, varNumCon, 5))
        # Array for deconvolution results in equi-volume space:
        aryDecon = np.zeros((varNumSub, varNumCon, varNumDpth))

    elif (varMdl == 4) or (varMdl == 5):
        # The array for single-subject deconvolution result has an additional
        # dimension in case of model 4 (number of iterations):
        aryDecon5 = np.zeros((varNumSub, varNumIt, varNumCon, 5))
        # Array for deconvolution results in equi-volume space:
        aryDecon = np.zeros((varNumSub, varNumIt, varNumCon, varNumDpth))
        # Generate random noise for model 4:
        aryNseRnd = np.random.randn(varNumIt, varNumCon, varNumDpth)
        # Scale variance:
        aryNseRnd = np.multiply(aryNseRnd, varNseRndSd)
        # Centre at one:
        aryNseRnd = np.add(aryNseRnd, 1.0)

    if varMdl == 5:
        # Additional array for deconvolution results with systematic error,
        # defined at 5 depth levels:
        arySys5 = np.zeros((varNumSub, 2, varNumCon, 5))
        # Array for deconvolutino results with systematic error, defined at
        # empirical depth levels:
        arySys = np.zeros((varNumSub, 2, varNumCon, varNumDpth))

    if varMdl == 6:
        # Array for single-subject deconvolution result (defined at 5 depth
        # levels):
        aryDecon5 = np.zeros((varNumSub, len(lstFctr), varNumCon, 5))
        # Array for deconvolution results in equi-volume space:
        aryDecon = np.zeros((varNumSub, len(lstFctr), varNumCon, varNumDpth))

    for idxSub in range(0, varNumSub):

        # ---------------------------------------------------------------------
        # *** Interpolation (downsampling)

        # The empirical depth profiles are defined at more depth levels than
        # the draining model. We downsample the empirical depth profiles to the
        # number of depth levels of the model.

        # The relative thickness of the layers differs between V1 & V2.
        if strRoi == 'v1':
            print('------Interpolation - V1')
            # Relative thickness of the layers (layer VI, 20%; layer V, 10%;
            # layer IV, 40%; layer II/III, 20%; layer I, 10%; Markuerkiaga et
            # al. 2016). lstThck = [0.2, 0.1, 0.4, 0.2, 0.1]
            # From the relative thickness, we derive the relative position of
            # the layers (we set the position of each layer to the sum of all
            # lower layers plus half  its own thickness):
            vecPosMdl = np.array([0.1, 0.25, 0.5, 0.8, 0.95])

        elif strRoi == 'v2':
            print('------Interpolation - V2')
            # Relative position of the layers (accordign to Weber et al., 2008,
            # Figure 5C, p. 2322). We start with the absolute depth:
            vecPosMdl = np.array([160.0, 590.0, 1110.0, 1400.0, 1620.0])
            # Divide by overall thickness (1.7 mm):
            vecPosMdl = np.divide(vecPosMdl, 1700.0)

        # Position of empirical datapoints:
        vecPosEmp = np.linspace(np.min(vecPosMdl),
                                np.max(vecPosMdl),
                                num=varNumDpth,
                                endpoint=True)

        # Vector for downsampled empirical depth profiles:
        aryEmp5 = np.zeros((varNumCon, 5))

        # Loop through conditions and downsample the depth profiles:
        for idxCon in range(0, varNumCon):
            # Interpolation:
            aryEmp5[idxCon] = griddata(vecPosEmp,
                                       aryEmpSnSb[idxSub, idxCon, :],
                                       vecPosMdl,
                                       method='cubic')

        # Put interpolation result for this subject into the array:
        aryEmp5SnSb[idxSub, :, :] = np.copy(aryEmp5)

        # ---------------------------------------------------------------------
        # *** Subtraction of draining effect

        # (1) Deconvolution based on Markuerkiaga et al. (2016).
        if varMdl == 1:
            aryDecon5[idxSub, :, :] = deconv_01(varNumCon,
                                                aryEmp5SnSb[idxSub, :, :])

        # (2) Deconvolution based on Markuerkiaga et al. (2016) & scaling based
        #     on Markuerkiaga et al. (2016).
        elif varMdl == 2:
            aryDecon5[idxSub, :, :] = deconv_02(varNumCon,
                                                aryEmp5SnSb[idxSub, :, :])

        # (3) Deconvolution based on Markuerkiaga et al. (2016) & scaling based
        #     on Weber et al. (2008).
        elif varMdl == 3:
            aryDecon5[idxSub, :, :] = deconv_03(varNumCon,
                                                aryEmp5SnSb[idxSub, :, :],
                                                strRoi=strRoi)

        # (4) Deconvolution based on Markuerkiaga et al. (2016), with random
        #     error.
        elif varMdl == 4:
                aryDecon5[idxSub, :, :, :] = deconv_04(varNumCon,
                                                       aryEmp5,
                                                       aryNseRnd)

        # (5) Deconvolution based on Markuerkiaga et al. (2016), with random
        #     and systematic error.
        elif varMdl == 5:
                aryDecon5[idxSub, :, :, :], arySys5[idxSub, :, :, :] = \
                    deconv_05(varNumCon, aryEmp5, aryNseRnd, varNseSys)

        # (6) Deconvolution based on Markuerkiaga et al. (2016), with deep GM
        #     signal scaling factor.
        elif varMdl == 6:
                aryDecon5[idxSub, :, :, :] = \
                    deconv_06(varNumCon, aryEmp5, lstFctr)

        # ---------------------------------------------------------------------
        # *** Interpolation

        # The original depth profiles were in 'equi-volume' space, and needed
        # to be downsampled in order to apply the deconvolution (because the
        # deconvolution model is defined at a lower number of depth levels than
        # the equivolume space). Here, the results of the deconvolution are
        # brought back into equivolume space. This is advantageous for the
        # creation of depth plots (equal spacing of data points on x-axis), and
        # for the calculation of peak positions (no additional information
        # about relative position of datapoints needs to be passed on).

        # Sampling points for equi-volume space:
        vecIntpEqui = np.linspace(np.min(vecPosMdl),
                                  np.max(vecPosMdl),
                                  num=varNumDpth,
                                  endpoint=True)

        if (varMdl != 4) and (varMdl != 5) and (varMdl != 6):

            # Loop through conditions:
            for idxCon in range(0, varNumCon):

                # Interpolation back into equi-volume space:
                aryDecon[idxSub, idxCon, :] = griddata(vecPosMdl,
                                                       aryDecon5[idxSub,
                                                                 idxCon,
                                                                 :],
                                                       vecIntpEqui,
                                                       method='cubic')

        elif (varMdl == 4) or (varMdl == 5):

            # Loop through iterations:
            for idxIt in range(0, varNumIt):

                # Loop through conditions:
                for idxCon in range(0, varNumCon):

                    # Interpolation back into equi-volume space:
                    aryDecon[idxSub, idxIt, idxCon, :] = \
                        griddata(vecPosMdl,
                                 aryDecon5[idxSub, idxIt, idxCon, :],
                                 vecIntpEqui,
                                 method='cubic')

        # For model 5, also resample systematic error term:
        if varMdl == 5:

            # Loop through conditions:
            for idxCon in range(0, varNumCon):

                # Interpolation for lower limit of systematic error term:
                arySys[idxSub, 0, idxCon, :] = \
                    griddata(vecPosMdl,
                             arySys5[idxSub, 0, idxCon, :],
                             vecIntpEqui,
                             method='cubic')

                # Interpolation for upper limit of systematic error term:
                arySys[idxSub, 1, idxCon, :] = \
                    griddata(vecPosMdl,
                             arySys5[idxSub, 1, idxCon, :],
                             vecIntpEqui,
                             method='cubic')

        # For model 6, loop through deep-GM-signal-intensity-scaling-factors:
        if varMdl == 6:

            # The array now has the form: aryDecon[idxSub, idxFctr, idxCon,
            # idxDepth], where idxFctr corresponds to the
            # deep-GM-signal-intensity-scaling-factors.

            # Loop through factors:
            for idxFctr in range(len(lstFctr)):

                # Loop through conditions:
                for idxCon in range(varNumCon):

                    # Interpolation back into equi-volume space:
                    aryDecon[idxSub, idxFctr, idxCon, :] = \
                        griddata(vecPosMdl,
                                 aryDecon5[idxSub, idxFctr, idxCon, :],
                                 vecIntpEqui,
                                 method='cubic')

    # -------------------------------------------------------------------------
    # *** Save corrected depth profiles

    if (varMdl != 4) and (varMdl != 5) and (varMdl != 6):
        # Save array with single-subject corrected depth profiles, of the form
        # aryDecon[idxSub, idxCondition, idxDpth].
        np.save(strPthPrfOt,
                aryDecon)

    # -------------------------------------------------------------------------
    # *** Peak positions percentile bootstrap

    # Bootstrapping in order to obtain an estimate of across-subjects variance
    # is not performed for models 4 & 5. (Models 4 & 5 are used to test the
    # effect of error in the model assumptions.)
    if (varMdl != 4) and (varMdl != 5) and (varMdl != 6):

        print('---Peak positions in depth profiles - percentile bootstrap')

        # We bootstrap the peak finding. Peak finding needs to be performed
        # both before and after deconvolution, separately for all stimulus
        # conditions.

        # Random array with subject indicies for bootstrapping of the form
        # aryRnd[varNumIt, varNumSmp]. Each row includes the indicies of the
        # subjects to the sampled on that iteration.
        aryRnd = np.random.randint(0,
                                   high=varNumSub,
                                   size=(varNumIt, varNumSub))

        # Loop before/after deconvolution:
        for idxDec in range(2):

            if idxDec == 0:
                print('------UNCORRECTED')

            if idxDec == 1:
                print('------CORRECTED')

            # Array for peak positins before deconvolution, of the form
            # aryPks01[idxCondition, idxIteration]
            aryPks01 = np.zeros((2, varNumCon, varNumIt))

            # Array for actual bootstrap samples:
            aryBoo = np.zeros((varNumIt, varNumDpth))

            # Loop through conditions:
            for idxCon in range(0, varNumCon):

                # Create array with bootstrap samples:
                for idxIt in range(0, varNumIt):

                    # Before deconvolution:
                    if idxDec == 0:
                        # Take mean across subjects in bootstrap samples:
                        aryBoo[idxIt, :] = np.mean(aryEmpSnSb[aryRnd[idxIt, :],
                                                              idxCon,
                                                              :],
                                                   axis=0)

                    # After deconvolution:
                    if idxDec == 1:
                        # Take mean across subjects in bootstrap samples:
                        aryBoo[idxIt, :] = np.mean(aryDecon[aryRnd[idxIt, :],
                                                            idxCon,
                                                            :],
                                                   axis=0)

                # Find peaks:
                aryPks01[idxDec, idxCon, :] = find_peak(aryBoo,
                                                        varNumIntp=100,
                                                        varSd=0.05,
                                                        lgcStat=False)

                # Median peak position:
                varTmpMed = np.median(aryPks01[idxDec, idxCon, :])

                # Confidence interval (percentile bootstrap):
                varTmpCnfLw = np.percentile(aryPks01[idxDec, idxCon, :],
                                            varCnfLw)
                varTmpCnfUp = np.percentile(aryPks01[idxDec, idxCon, :],
                                            varCnfUp)

                # Print result:
                strTmp = ('---------Median peak position: '
                          + str(np.around(varTmpMed, decimals=2)))
                print(strTmp)
                strTmp = ('---------Percentile bootstrap '
                          + str(np.around(varCnfLw, decimals=1))
                          + '%: '
                          + str(np.around(varTmpCnfLw, decimals=2)))
                print(strTmp)
                strTmp = ('---------Percentile bootstrap '
                          + str(np.around(varCnfUp, decimals=1))
                          + '%: '
                          + str(np.around(varTmpCnfUp, decimals=2)))
                print(strTmp)

    # -------------------------------------------------------------------------
    # *** Plot results

    print('---Plot results')

    if (varMdl != 4) and (varMdl != 5) and (varMdl != 6):

        # Plot across-subjects mean before deconvolution:
        strTmpTtl = '{} before deconvolution'.format(strRoi.upper())
        strTmpPth = (strPthPltOt + 'before_')
        plt_dpth_prfl_acr_subs(aryEmpSnSb,
                               varNumSub,
                               varNumDpth,
                               varNumCon,
                               varDpi,
                               varAcrSubsYmin01,
                               varAcrSubsYmax01,
                               lstConLbl,
                               strXlabel,
                               strYlabel,
                               strTmpTtl,
                               strTmpPth,
                               strFlTp,
                               strErr='sem',
                               vecX=vecPosEmp)

        # Across-subjects mean after deconvolution:
        strTmpTtl = '{} after deconvolution'.format(strRoi.upper())
        strTmpPth = (strPthPltOt + 'after_')
        plt_dpth_prfl_acr_subs(aryDecon,
                               varNumSub,
                               varNumDpth,
                               varNumCon,
                               varDpi,
                               varAcrSubsYmin02,
                               varAcrSubsYmax02,
                               lstConLbl,
                               strXlabel,
                               strYlabel,
                               strTmpTtl,
                               strTmpPth,
                               strFlTp,
                               strErr='sem',
                               vecX=vecIntpEqui)

    elif varMdl == 4:

        # For 'model 4', i.e. the random noise model, we are interested in the
        # variance across random-noise iterations. We are *not* interested in
        # the variance across subjects in this case. Because we used the same
        # random noise across subjects, we can average over subjects.
        aryDecon = np.mean(aryDecon, axis=0)

        # Across-subjects mean after deconvolution:
        strTmpTtl = '{} after deconvolution'.format(strRoi.upper())
        strTmpPth = (strPthPltOt + 'after_')
        plt_dpth_prfl_acr_subs(aryDecon,
                               varNumSub,
                               varNumDpth,
                               varNumCon,
                               varDpi,
                               varAcrSubsYmin02,
                               varAcrSubsYmax02,
                               lstConLbl,
                               strXlabel,
                               strYlabel,
                               strTmpTtl,
                               strTmpPth,
                               strFlTp,
                               strErr='prct95',
                               vecX=vecIntpEqui)

    elif varMdl == 5:

        # For 'model 5', i.e. the random & systematic noise model, we are
        # interested in the variance across random-noise iterations. We are
        # *not* interested in the variance across subjects in this case.
        # Because we used the same random noise across subjects, we can average
        # over subjects.
        aryDecon = np.mean(aryDecon, axis=0)

        # Random noise - mean across iteratins:
        aryRndMne = np.mean(aryDecon, axis=0)
        # Random noise -  lower percentile:
        aryRndConfLw = np.percentile(aryDecon, varCnfLw, axis=0)
        # Random noise - upper percentile:
        aryRndConfUp = np.percentile(aryDecon, varCnfUp, axis=0)

        # For model 5, we only plot one stimulus condition (condition 4):
        varTmpCon = 3
        aryRndMne = aryRndMne[varTmpCon, :]
        aryRndConfLw = aryRndConfLw[varTmpCon, :]
        aryRndConfUp = aryRndConfUp[varTmpCon, :]

        # Systematic error term - mean across subjects:
        arySysMne = np.mean(arySys, axis=0)

        # Patching together systematic and random error terms:
        aryComb = np.array([aryRndMne,
                            arySysMne[0, varTmpCon, :],
                            arySysMne[1, varTmpCon, :]])

        # aryRndMne.shape
        # arySysMne[0, varTmpCon, :].shape
        # arySysMne[1, varTmpCon, :].shape

        # Patching together array for error shading (no shading for systematic
        # error term):
        aryErrLw = np.array([aryRndConfLw,
                             arySysMne[0, varTmpCon, :],
                             arySysMne[1, varTmpCon, :]])
        aryErrUp = np.array([aryRndConfUp,
                             arySysMne[0, varTmpCon, :],
                             arySysMne[1, varTmpCon, :]])

        # *** Plot response at half maximum contrast across depth

        strTmpTtl = '{}'.format(strRoi.upper())
        strTmpPth = (strPthPltOt + 'after_')

        # Labels for model 5:
        lstLblMdl5 = ['Random error',
                      'Systematic error',
                      'Systematic error']

        # Label for axes:
        strXlabel = 'Cortical depth level (equivolume)'
        strYlabel = 'fMRI signal change [a.u.]'

        # Colour for systematic error plot:
        aryClr = np.array(([22.0, 41.0, 248.0],
                           [230.0, 56.0, 60.0],
                           [230.0, 56.0, 60.0]))
        aryClr = np.divide(aryClr, 255.0)

        plt_acr_dpth(aryComb,        # aryData[Condition, Depth]
                     0,              # aryError[Con., Depth]
                     varNumDpth,     # Number of depth levels (on the x-axis)
                     3,              # Number of conditions (separate lines)
                     varDpi,         # Resolution of the output figure
                     0.0,            # Minimum of Y axis
                     2.0,            # Maximum of Y axis
                     False,          # Bool.: whether to convert y axis to %
                     lstLblMdl5,     # Labels for conditions (separate lines)
                     strXlabel,      # Label on x axis
                     strYlabel,      # Label on y axis
                     strTmpTtl,      # Figure title
                     True,           # Boolean: whether to plot a legend
                     (strPthPltOt + 'after' + strFlTp),
                     varSizeX=2000.0,
                     varSizeY=1400.0,
                     aryCnfLw=aryErrLw,
                     aryCnfUp=aryErrUp,
                     aryClr=aryClr,
                     vecX=vecIntpEqui)

    elif varMdl == 6:

        # The array now has the form: aryDecon[idxSub, idxFctr, idxCon,
        # idxDepth], where idxFctr corresponds to the
        # deep-GM-signal-intensity-scaling-factors.

        # For 'model 6', i.e. the deep-GM signal underestimation model, we are
        # *not* interested in the variance across subjects, but in the effect
        # of the deep-GM signal scaling factor. Because we used the same
        # deep-GM scaling factor across subjects, we can average over subjects.
        aryDecon = np.mean(aryDecon, axis=0)

        # The array now has the form: aryDecon[idxFctr, idxCon, idxDepth],
        # where idxFctr corresponds to the
        # deep-GM-signal-intensity-scaling-factors.

        # Reduce further; only one stimulus condition is plotted. The
        # deep-GM-signal-intensity-scaling-factors are treated as conditions
        # for the plot.
        aryDecon = aryDecon[:, 3, :]

        # The array now has the form: aryDecon[idxFctr, idxDepth], where
        # idxFctr corresponds to the deep-GM-signal-intensity-scaling-factors.

        # Dummy error array (no error will be plotted):
        aryErr = np.zeros(aryDecon.shape)

        strTmpTtl = '{}'.format(strRoi.upper())
        strTmpPth = (strPthPltOt + 'after_')

        # Labels for model 6 (deep-GM-signal-intensity-scaling-factors):
        lstLblMdl5 = [(str(int(np.around(x * 100.0))) + ' %') for x in lstFctr]

        # Label for axes:
        strXlabel = 'Cortical depth level (equivolume)'
        strYlabel = 'fMRI signal change [a.u.]'

        plt_acr_dpth(aryDecon,           # aryData[Condition, Depth]
                     aryErr,             # aryError[Con., Depth]
                     varNumDpth,         # Number of depth levels (on x-axis)
                     aryDecon.shape[0],  # Number conditions (separate lines)
                     varDpi,             # Resolution of the output figure
                     0.0,                # Minimum of Y axis
                     2.0,                # Maximum of Y axis
                     False,              # Bool: convert y axis to % ?
                     lstLblMdl5,         # Condition labels (separate lines)
                     strXlabel,          # Label on x axis
                     strYlabel,          # Label on y axis
                     strTmpTtl,          # Figure title
                     True,               # Boolean: whether to plot a legend
                     (strPthPltOt + 'after' + strFlTp),
                     varSizeX=2000.0,
                     varSizeY=1400.0,
                     vecX=vecIntpEqui)

    # -------------------------------------------------------------------------
    print('-Done.')
