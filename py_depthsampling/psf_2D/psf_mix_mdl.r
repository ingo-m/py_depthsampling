# Mixed regression on parameters of cortical depth point spread function.
#
# Data is loaded from csv file containing dataframe. Dataframe csv file can be
# created using `py_depthsampling.psf_2D.psf_2D_main.py`.

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

# Requires:
# install.packages('lme4')
# library(lme4)

# Path of input csv file:
strCsv <- '/home/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf_2D/dataframe.csv'

# Read data from disk:
dfPsf = read.csv(strCsv)

colnames(dfPsf)

# Create mixed effects model:
# objModel <- lmer(Width ~ ROI + Condition + (1|Depth), data=dfPsf)

objModel <- lm(Width ~ ROI + Condition + Depth, data=dfPsf)

# objModel <- aov(Width ~ ROI + Condition + Depth, data=dfPsf)

summary(objModel)

