###########################
# ANOVA for effect of ROI #
###########################

# Perform an ANOVA for an effect of ROI separately at deep or middle or
# superficial cortical depth.

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_stimulus.csv', header=TRUE, sep=';')
# objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_periphery.csv', header=TRUE, sep=';')

# Test only PacMan dynamic and control dynamic conditions, exclude PacMan static
# condition:
objDf <- objDf[objDf$Condition!='PS',]

# Test only two ROIs (e.g. V1 and V2):
objDf <- objDf[objDf$ROI!='V3',]

# Test only the deepest cortical depth levels:
objDf <- objDf[objDf$Depth<3,]

# Dimensionality reduction - mean over depth levels:
objDf <- aggregate(PSC ~ ROI + Condition + Subject, objDf, mean)

# Fit ANOVA model:
objAov <-  aov(PSC ~ ROI*Condition + Subject, data=objDf)
# Weighting?

summary(objAov)
