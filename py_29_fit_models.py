# -*- coding: utf-8 -*-


"""
The purpose of this script is to plot measured data and to fit models to that
data. One use is, for instance, the plotting of luminance versus pixel
intensity values as measured on the 7T projector.
(C) Ingo Marquardt, 25.05.2016
"""


# *****************************************************************************
# ***Load modules

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# *****************************************************************************


# *****************************************************************************
# *** Data to plot

# Independent variable data:
vecInd = np.array([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0,
                   -0.125, -0.25, -0.375, -0.5, -0.625, -0.75, -0.875, -1.0])

# Dependent variable (measured data), in separat rows for repetitoins of the
# measurement:
vecDep = np.array([[475.0, 434.0, 402.0, 362.0, 323.0, 281.0, 240.0, 201.0,
                    161.0, 121.0, 85.5, 57.8, 33.3, 15.6, 6.0, 1.5, 0.76],
                   [488.0, 459.0, 423.0, 384.0, 341.0, 295.0, 252.0, 214.0,
                    170.0, 127.0, 91.4, 51.5, 36.5, 15.6, 5.4, 1.28, 0.76]])

# Label for x-axis (independent variable):
strLblX = 'Pixel value [-1 to 1]'

# Label for y-axis (dependent variable):
strLblY = 'Luminance [cd/m^2]'

# Figure title:
strTlt = 'Luminance as a function of psychopy pixel intensity at 7T projector'

# Limits of x-axis:
vecXlim = [-1.1, 1.1]

# Limits of y-axis:
vecYlim = [-10.0, 500.0]

# Output directory for figures:
strPathOut = '/home/john/Desktop/'
# *****************************************************************************


# *****************************************************************************
# *** Functions

# Define exponential function to be fitted to the measurement data:
def funcExp(varX, varA, varB, varC):
    varOut = varA * np.exp(-varB * varX) + varC
    return varOut


# Define logarithmic function to be fitted to the measurement data:
def funcLn(varX, varA, varB):
    varOut = varA * np.log(varX) + varB
    return varOut


# Define 2nd degree polynomial function to be fitted to the measurement data:
def funcPoly2(varX, varA, varB, varC):
    varOut = (varA * np.power(varX, 2) +
              varB * np.power(varX, 1) +
              varC)
    return varOut


# Define 3rd degree polynomial function to be fitted to the measurement data:
def funcPoly3(varX, varA, varB, varC, varD):
    varOut = (varA * np.power(varX, 3) +
              varB * np.power(varX, 2) +
              varC * np.power(varX, 1) +
              varD)
    return varOut


# Define power function to be fitted to the measurement data:
def funcPow(varX, varA, varB, varC, varD):
    varOut = (varA * np.power((varX + varB), varC) + varD)
    return varOut
# *****************************************************************************


# *****************************************************************************
# *** Preparations

# Calculate average of dependent variable:
vecDepAvg = np.mean(vecDep, axis=0)

# Calculate standard deviations of dependent variable:
vecStd = np.std(vecDep, axis=0)
# *****************************************************************************


# *****************************************************************************
#  Exponential model fitting:

# Fit the model to the exponential function:
vecExpModelPar, vecExpModelCov = curve_fit(funcExp, vecInd, vecDepAvg)

# Calculate fitted values:
vecFittedExp = funcExp(vecInd,
                       vecExpModelPar[0],
                       vecExpModelPar[1],
                       vecExpModelPar[2])

# Create string for model parameters of exponential function:
varTmpA = np.around(vecExpModelPar[0], 0)
varTmpB = np.around(vecExpModelPar[1], 2)
varTmpC = np.around(vecExpModelPar[2], 0)
strModelExp = 'y = ' + \
              str(varTmpA) + \
              ' e ^ ( -' + \
              str(varTmpB) + \
              ' * x ) + ' + \
              str(varTmpC)
# *****************************************************************************


# *****************************************************************************
# *** Logarithmic model fitting

# Fit the model to the logarithmic function:
vecLnModelPar, vecLnModelCov = curve_fit(funcLn, vecInd, vecDepAvg)

# Calculate fitted values:
vecFittedLn = funcLn(vecInd,
                     vecLnModelPar[0],
                     vecLnModelPar[1])

# Create string for model parameters of exponential function:
varTmpA = np.around(vecLnModelPar[0], 0)
varTmpB = np.around(vecLnModelPar[1], 0)
strModelLn = 'y = ' + \
             str(varTmpA) + \
             ' * ln(x) + ' + \
             str(varTmpB)
# *****************************************************************************


# *****************************************************************************
# *** Polynomial model fitting 2nd degree

# Fit the model to the 2nd degree polynomial function:
vecPoly2ModelPar, vecPoly2ModelCov = curve_fit(funcPoly2, vecInd, vecDepAvg)

# Calculate fitted values:
vecFittedPoly2 = funcPoly2(vecInd,
                           vecPoly2ModelPar[0],
                           vecPoly2ModelPar[1],
                           vecPoly2ModelPar[2])

# Create string for model parameters of exponential function:
varTmpA = np.around(vecPoly2ModelPar[0], 2)
varTmpB = np.around(vecPoly2ModelPar[1], 2)
varTmpC = np.around(vecPoly2ModelPar[2], 2)
strModelPoly2 = 'y = ' + \
                str(varTmpA) + \
                ' * x^2 + ' + \
                str(varTmpB) + \
                ' * x^1 + ' + \
                str(varTmpC)
# *****************************************************************************


# *****************************************************************************
# *** Polynomial model fitting 3rd degree

# Fit the model to the 3rd degree polynomial function:
vecPoly3ModelPar, vecPoly3ModelCov = curve_fit(funcPoly3, vecInd, vecDepAvg)

# Calculate fitted values:
vecFittedPoly3 = funcPoly3(vecInd,
                           vecPoly3ModelPar[0],
                           vecPoly3ModelPar[1],
                           vecPoly3ModelPar[2],
                           vecPoly3ModelPar[3])

# Create string for model parameters of exponential function:
varTmpA = np.around(vecPoly3ModelPar[0], 1)
varTmpB = np.around(vecPoly3ModelPar[1], 1)
varTmpC = np.around(vecPoly3ModelPar[2], 1)
varTmpD = np.around(vecPoly3ModelPar[3], 1)
strModelPoly3 = 'y = ' + \
                str(varTmpA) + \
                ' * x^3 + ' + \
                str(varTmpB) + \
                ' * x^2 + ' + \
                str(varTmpC) + \
                ' * x + ' + \
                str(varTmpD)
# *****************************************************************************


# *****************************************************************************
# *** Power function fitting

# Fit the model to the 3rd degree polynomial function:
vecPowModelPar, vecPowModelCov = curve_fit(funcPow, vecInd, vecDepAvg)

# Calculate fitted values:
vecFittedPow = funcPow(vecInd,
                       vecPowModelPar[0],
                       vecPowModelPar[1],
                       vecPowModelPar[2],
                       vecPowModelPar[3])

# Create string for model parameters of exponential function:
varTmpA = np.around(vecPowModelPar[0], 1)
varTmpB = np.around(vecPowModelPar[1], 1)
varTmpC = np.around(vecPowModelPar[2], 1)
varTmpD = np.around(vecPowModelPar[3], 1)
strModelPow = 'y = ' + \
              str(varTmpA) + \
              ' * (x + ' + \
              str(varTmpB) + \
              ') ^ ' + \
              str(varTmpC) + \
              ' + ' + \
              str(varTmpD)
# *****************************************************************************


# *****************************************************************************
# *** Create plots

# List with model predictions:
lstModPre = [vecFittedExp,
             vecFittedLn,
             vecFittedPoly2,
             vecFittedPoly3,
             vecFittedPow]

# List with model parameters:
lstModPar = [strModelExp,
             strModelLn,
             strModelPoly2,
             strModelPoly3,
             strModelPow]

# We create one plot per function:
for idxPlt in range(0, len(lstModPre)):

    fig01 = plt.figure()

    axs01 = fig01.add_subplot(111)

    # Plot the average dependent data with error bars:
    plt01 = axs01.errorbar(vecInd,
                           vecDepAvg,
                           yerr=vecStd,
                           color='blue',
                           label='Mean (SD)',
                           linewidth=0.9,
                           antialiased=True)

    # Plot model prediction:
    plt02 = axs01.plot(vecInd,
                       lstModPre[idxPlt],
                       color='red',
                       label=lstModPar[idxPlt],
                       linewidth=1.0,
                       antialiased=True)

    # Limits of the x-axis:
    # axs01.set_xlim([np.min(vecInd), np.max(vecInd)])
    axs01.set_xlim([vecXlim[0], vecXlim[1]])

    # Limits of the y-axis:
    axs01.set_ylim([vecYlim[0], vecYlim[1]])

    # Adjust labels for axis 1:
    axs01.tick_params(labelsize=10)
    axs01.set_xlabel(strLblX, fontsize=9)
    axs01.set_ylabel(strLblY, fontsize=9)
    axs01.set_title(strTlt, fontsize=9)

    # Add legend:
    axs01.legend(loc=0, prop={'size': 9})

    # Add vertical grid lines:
    axs01.xaxis.grid(which=u'major',
                     color=([0.5, 0.5, 0.5]),
                     linestyle='-',
                     linewidth=0.3)

    # Add horizontal grid lines:
    axs01.yaxis.grid(which=u'major',
                     color=([0.5, 0.5, 0.5]),
                     linestyle='-',
                     linewidth=0.3)

    # Save figure:
    fig01.savefig((strPathOut + 'plot_' + str(idxPlt) + '.png'),
                  dpi=200,
                  facecolor='w',
                  edgecolor='w',
                  orientation='landscape',
                  papertype='a6',
                  transparent=False,
                  frameon=None)
# *****************************************************************************
