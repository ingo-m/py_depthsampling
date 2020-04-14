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
from scipy.stats import norm
from py_depthsampling.ert.ert_plt import ert_plt


# -----------------------------------------------------------------------------
# ***  Define parameters

# Output path for plots (file name left open):
strPthOut = '/home/john/Dropbox/University/PhD/PacMan_Project/Figures/F04_S04_Timecourse_simulation_REVISION_02/elements/{}.png'


# -----------------------------------------------------------------------------
# *** Create model of positive sustained response

# Surface stimulus duration:
srf_stim_dur = 400

# Additional duration of texture background (texture background will be so much
# longer than surface stimulus):
txtr_stim_dur_add = 300

# Boxcar timecourse:
vecBox = np.concatenate([np.ones(srf_stim_dur),
                         (np.ones(srf_stim_dur) * -0.3)])

# Create 1D Gaussian:
mu = 0.0
sigma = 0.3
x_lin = np.linspace(-1.0, 3.0, num=len(vecBox))
gaussian = norm.pdf(x_lin, mu, sigma)

# Sustained positive fmri response:
vecFmri01 = np.convolve(vecBox, gaussian)[:1200]

# Normalise amplitude:
y_max = 3.0
vecFmri01 = np.divide(vecFmri01, np.max(vecFmri01)) * y_max

# Make positive plateau a bit longer:
x_argmax = np.argmax(vecFmri01)
vecFmri01 = np.concatenate([vecFmri01[:x_argmax],
                            (np.ones(txtr_stim_dur_add) * y_max),
                            vecFmri01[x_argmax:]])


# -----------------------------------------------------------------------------
# *** Response to texture background

# The response to texture background has a positive amplitude of ~4%, and
# causes an elevated baseline. Let's assume the surface stimulus onset is at
# timepoint x, when the elevated baseline response is ~4%
srf_stim_onset = 700

# vecFmri01[srf_stim_onset]

# Concatenate two background texture response (before and after surface
# stimulus):
vecFmriTxtr = np.concatenate([vecFmri01, vecFmri01])

# Let's assume that the response to the uniform surface has an amplitude of
# ~2%, and onset at timepoint 670.
vecFmri02 = vecFmri01 * 0.3
vecFmri02 = np.concatenate([np.zeros(srf_stim_onset),
                            vecFmri02])
# Concatenate zeors to surface response to reach same length as texture
# response:
vecFmriSrf = np.concatenate([vecFmri02,
                             np.zeros(len(vecFmriTxtr) - len(vecFmri02))])


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
