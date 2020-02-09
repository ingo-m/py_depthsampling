
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_stimulus.csv', header=TRUE, sep=';')
# objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data_periphery.csv', header=TRUE, sep=';')

# head(objDf)

# Test only PacMan dynamic and control dynamic conditions, exclude PacMan static
# condition:
# objDf <- objDf[objDf$Condition!='PS',]

# Test only the deepest cortical depth levels:
# objDf <- objDf[objDf$Depth<3,]

# Test only V2 and V3:
objDf <- objDf[objDf$ROI!='V1',]

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

# (3)
# Is there an effect of condition on the depth profiles with respect to
# condition (i.e. do the condition differences over cortical depth differ
#between ROI)?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth +
              Condition:Depth:ROI, # Effect of interest
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
