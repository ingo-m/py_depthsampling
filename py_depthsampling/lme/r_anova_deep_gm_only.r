###########################
# ANOVA for effect of ROI #
###########################

# Only needed for regression model (see below).
library(nlme)

# (1)
# Perform an ANOVA for an effect of ROI separately at deep or middle or
# superficial cortical depth.

# Read CSV into R:
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_stimulus_nvertices.csv',
                  header=TRUE,
                  sep=';')

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
objAov <-  aov(PSC ~ ROI * Condition + Subject,
               data=objDf,
               weights=objDf$Vertices)

summary(objAov)

# (2)
# Linear mixed model comparison.

# Full model:
mdlFull = lme(PSC ~ ROI + Condition +
              ROI:Condition,  # Effect of interest
              objDf,
              random=(~1|Subject),
              method='ML')

# Null model:
mdlNull = lme(PSC ~ ROI + Condition,
              objDf,
              random=(~1|Subject),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)
