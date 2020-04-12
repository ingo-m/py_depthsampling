#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate composite positive and negative fMRI response.

Simulation to address reviewer comment (2), second revision round, PacMan paper
eLife submission.

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


# -----------------------------------------------------------------------------
# *** Create model of positive sustained response

# Boxcar timecourse:
vecBox = np.concatenate([np.ones(400), (np.ones(400) * -0.3)])

# Create 1D Gaussian:
mu = 0.0
sigma = 0.3
x_lin = np.linspace(-1.0, 3.0, num=len(vecBox))
gaussian = norm.pdf(x_lin, mu, sigma)

# Sustained positive fmri response:
vecFmri01 = np.convolve(vecBox, gaussian)[:1200]

# Normalise amplitude:
y_max = 4.0
vecFmri01 = np.divide(vecFmri01, np.max(vecFmri01)) * y_max

# Make positive plateau a bit longer:
x_argmax = np.argmax(vecFmri01)
vecFmri01 = np.concatenate([vecFmri01[:x_argmax],
                            (np.ones(300) * y_max),
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

aaa = np.array([vecFmriTxtr, vecFmriSrf]).T

aaa = np.add(vecFmriTxtr, vecFmriSrf)

