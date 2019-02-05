
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_stimulus.csv', header=TRUE, sep=';')
# objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_periphery.csv', header=TRUE, sep=';')

# head(objDf)

# (1)
# Do the stimuli differentially activate the ROIs (i.e. does activation differ
# between ROIs as a function of condition)?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +  # Effect of interest
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (2)
# Do the stimuli case differential activation across cortical depth levels (i.e.
# does activation differ between depth levels as a function of condition)?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +  # Effect of interest
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# # Linear model using generalised least squares:
# mdl_gls = gls(PSC ~ ROI * Condition * Depth * Subject,
#               objDf,
#               correlation=corCAR1(form=(~1|Depth)),
#               method='ML')
#
# summary(mdl_gls)
#
# nlme:::summary.gls(mdl_gls)$tTable

#library(ggplot2)
#ggplot(data=objDf, aes(x=Depth, y=PSC, group=ROI, color=ROI)) +
#  geom_line()
