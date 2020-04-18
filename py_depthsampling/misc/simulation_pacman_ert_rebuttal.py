#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate composite positive and negative fMRI response.

Simulation to address reviewer comment (2), second revision round, PacMan paper
eLife submission.

TODO
- Stimulus duration (grey bar) is too short.

Reviewer comment:

> 2. Negative BOLD response due to stimulus with textured background.
> We remain puzzled by the negative BOLD. Intriguingly, the onset of a whole-
> field textured background (Fig6-S3) elicits an fMRI response that is
> comparable in positive amplitude to the negative effects within the stimulus
> region (Fig4&7). This might indicate that in the main experiment, the
> textured background causes ongoing positive responses throughout the run,
> presumably maintained by ongoing micro saccades and fixational instability.
> Then, the presentation onset of a stimulus has a two-fold effect; it
> decreases the ongoing response to this background (because it disappears)
> while also playing the role of an activating stimulus very locally (dark
> patches in the texture are brighter, whereas light patches are darker,
> stimulating both on and off visual channels temporarily). In this scenario,
> the negative response shown in Fig4 should be seen as a combination of a
> positive and a negative component, that summed together produce the
> appearance of a delayed negative response.
"""


import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import norm
from py_depthsampling.ert.ert_plt import ert_plt


# -----------------------------------------------------------------------------
# ***  Define parameters

# Output path for plots (file name left open):
strPthOut = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Project/Figures/F04_S04_Timecourse_simulation_REVISION_02/elements/{}.png'


# -----------------------------------------------------------------------------
# *** Parameters of texture response

# Amplitude of texture response:
y_max_txtr = 4.0

# Amplitude of post-stimulus undershoot:
y_min_txtr = -1.0

# Number of timepoints from onset to maximum amplitude:
dur_rise_txtr = 100

# Duration of plateau of texture response:
dur_max_txtr = 620

# Duration of fall from maximum to minimum (i.e. timepoints the signal takes to
# fall from maximum positive amplitude to minimum post-stimulus undershoot).
dur_fall_txtr = ((y_max_txtr + y_min_txtr) / y_max_txtr) * float(dur_rise_txtr)
dur_fall_txtr = int(np.around(dur_fall_txtr))

# Duration of post-stimulus undershoot of texture response:
dur_pstundr_txtr = int(np.around((0.25 * float(dur_max_txtr))))

# Duration of return to baseline after post-stimulus undershoot:
dur_rtrn_txtr = 50  # int(np.around((0.25 * float(dur_pstundr_txtr))))


# -----------------------------------------------------------------------------
# *** Construct texture response

# Response component 01 - rise:
vecFmri01 = np.linspace(0.0, y_max_txtr, num=dur_rise_txtr, endpoint=True)

# Response component 02 - plateau:
vecFmri02 = np.multiply(np.ones(dur_max_txtr), y_max_txtr)

# Response component 03 - fall:
vecFmri03 = np.linspace(y_max_txtr, y_min_txtr, num=dur_fall_txtr,
                        endpoint=True)

# Response component 04 - post-stimulus undershoot:
vecFmri04 = np.multiply(np.ones(dur_pstundr_txtr), y_min_txtr)

# Response component 05 - return to baseline:
vecFmri05 = np.linspace(y_min_txtr, 0.0, num=dur_rtrn_txtr, endpoint=True)


vecFmriTxtr = np.concatenate([vecFmri01,
                              vecFmri02,
                              vecFmri03,
                              vecFmri04,
                              vecFmri05])


# -----------------------------------------------------------------------------
# *** Parameters of surface response

# Amplitude of surface response:
y_max_srf = 1.0

# Amplitude of post-stimulus undershoot:
y_min_srf = -0.25

# Number of timepoints from onset to maximum amplitude. Same slope as texture
# response.
dur_rise_srf = (y_max_srf / y_max_txtr) * float(dur_rise_txtr)
dur_rise_srf = int(np.around(dur_rise_srf))

# Duration of plateau of surface response:
dur_max_srf = 220

# Duration of fall from maximum to minimum (i.e. timepoints the signal takes to
# fall from maximum positive amplitude to minimum post-stimulus undershoot).
dur_fall_srf = ((y_max_srf + y_min_srf) / y_max_srf) * float(dur_rise_srf)
dur_fall_srf = int(np.around(dur_fall_srf))

# Duration of post-stimulus undershoot of texture response:
dur_pstundr_srf = int(np.around((0.25 * float(dur_max_srf))))

# Duration of return to baseline after post-stimulus undershoot. Same slope as
# texture response.
dur_rtrn_srf = # TODO: IMPLEMENT SAME SLOPE


# -----------------------------------------------------------------------------
# *** Construct surface response

# Response component 01 - rise:
vecFmri01 = np.linspace(0.0, y_max_srf, num=dur_rise_srf, endpoint=True)

# Response component 02 - plateau:
vecFmri02 = np.multiply(np.ones(dur_max_srf), y_max_srf)

# Response component 03 - fall:
vecFmri03 = np.linspace(y_max_srf, y_min_srf, num=dur_fall_srf,
                        endpoint=True)

# Response component 04 - post-stimulus undershoot:
vecFmri04 = np.multiply(np.ones(dur_pstundr_srf), y_min_srf)

# Response component 05 - return to baseline:
vecFmri05 = np.linspace(y_min_srf, 0.0, num=dur_rtrn_srf, endpoint=True)


vecFmriSrf = np.concatenate([vecFmri01,
                             vecFmri02,
                             vecFmri03,
                             vecFmri04,
                             vecFmri05])









sigma = 20.0

aaa = gaussian_filter1d(vecFmriTxtr, sigma, mode='nearest')



# TODO:
# Surface timecourse as an inverted, shifted, scaled version of texture
# timecourse. Try manually creating shape, only use some smoothing for better
# visual appeal.

# -----------------------------------------------------------------------------
# *** Plot 1 - texture & surface response separately

# Dummy TR and scaling factor (to account for high-resolution timecourse,
# compared with empirical fMRI data).
varTr = 0.02
varTmeScl = 1.0 / float(varTr)

# Stimulus onset and duration, scaled (for plot axis labels):
srf_stim_onset_scl = float(srf_stim_onset) / varTmeScl
srf_stim_dur_scl = (float(srf_stim_dur) + float(srf_stim_onset)) / varTmeScl

# Plot labels:
lstConLbl = ['Texture background', 'Surface stimulus']
strXlabel = 'Time [s]'
strYlabel = 'Percent signal change'
strTitle = 'Schematic'

# Plot output path:
strPthOutTmp = strPthOut.format('plot_01_separate')

# Merge arrays of texture and surface responses for plot:
aryPlot01 = np.array([vecFmriTxtr, vecFmriSrf])

# Dummy array for error shading:
aryError01 = np.zeros(aryPlot01.shape)

# Number of 'volumes' (needed for axis labels):
varNumVol = aryPlot01.shape[1]

# Create plot:
ert_plt(aryPlot01,
        aryError01,
        None,  # Number of depth levels
        2,  # Number of conditions
        varNumVol,  # Number of volumes
        70.0,  # DPI
        -1.0,  # y axis minimum
        3.0,  # y axis maximum
        srf_stim_onset_scl,  # Pre-stimulus interval
        srf_stim_dur_scl,  # Stimulu end
        1.0,  # TR
        lstConLbl,
        True,  # Show legend?
        strXlabel,
        strYlabel,
        False,  # Convert to percent?
        strTitle,
        strPthOutTmp,
        varTmeScl=varTmeScl,
        varXlbl=10,
        varYnum=5,
        tplPadY=(0.5, 0.1),
        lstVrt=None,
        lstClr=None,
        lstLne=None)


# -----------------------------------------------------------------------------
# *** Calculate composite response

# Calculate composite response to texture and surface (as it would be observed
# empirically), with an elevated baseline (i.e. texture response as baseline).

# Add texture and surface responses:
vecFmriComp = np.add(vecFmriTxtr, vecFmriSrf)

# The pre-stimulus interval serves as baseline. Subtract the texture response
# amplitude to make the pre-stimulus response zero.
vecFmriComp = np.subtract(vecFmriComp, y_max)


# -----------------------------------------------------------------------------
# *** Plot composite response

# First dimension needs to be 'condition' for plot function:
aryPlot02 = np.array(vecFmriComp, ndmin=2)

# Plot output path:
strPthOutTmp = strPthOut.format('plot_02_composite')

ert_plt(aryPlot02,
        np.zeros(aryPlot02.shape),
        None,  # Number of depth levels
        1,  # Number of conditions
        varNumVol,  # Number of volumes
        70.0,  # DPI
        -4.0,  # y axis minimum
        0.0,  # y axis maximum
        srf_stim_onset_scl,  # Pre-stimulus interval
        srf_stim_dur_scl,  # Stimulu end
        1.0,  # TR
        lstConLbl,
        True,  # Show legend?
        strXlabel,
        strYlabel,
        False,  # Convert to percent?
        strTitle,
        strPthOutTmp,
        varTmeScl=varTmeScl,
        varXlbl=10,
        varYnum=5,
        tplPadY=(0.1, 0.1),
        lstVrt=None,
        lstClr=None,
        lstLne=None)
