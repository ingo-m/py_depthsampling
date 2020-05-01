#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot cortical depth profiles separately for early & late response.

Analysis to address reviewer comment (2), second revision round, PacMan paper
eLife submission.

Reviewer comment:

> 2. [...] If the negative BOLD response is indeed a composite of a positive
> and negative BOLD response, it would be interesting to see how the laminar
> effects can perhaps decompose this composite. A laminar analysis conducted on
> the first and later parts of the response separately may be highly insightful
> here.

"""


import pickle
import numpy as np


# -----------------------------------------------------------------------------
# *** Define parameters

# Path of pickle files with event-related timecourses (ROI left open):
pathPic = '/media/ssd_dropbox/Dropbox/University/PhD/PacMan_Depth_Data/Higher_Level_Analysis/stimulus/era_{}_rh.pickle'

# List of ROIs:
lstRois = ['v1', 'v2', 'v3']

# ...


# -----------------------------------------------------------------------------
# *** Loop through ROIs

for strRoi in lstRois:

    # Load pickle with event-related timecourses of current ROI:
    dicAllSubsRoiErt = pickle.load(open(pathPic.format(strRoi), 'rb'))


