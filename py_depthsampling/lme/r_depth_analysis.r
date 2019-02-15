
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/surface_depth_data_centre.csv', header=TRUE, sep=';')

head(objDf)

# Weights = ~1/n where n is number of vertices.

# (1) - Two-way interaction
# Do the stimuli differentially activate the ROIs (i.e. does activation differ
# between ROIs as a function of condition)?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +  # Effect of interest
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (2) - Two-way interaction
# Do the stimuli differentially activate the cortical depth levels (i.e. does
# activation differ between depth levels as a function of condition)?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth,  # Effect of interest
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (3) Three-way interaction
# Do the condition differences over cortical depth differ between ROIs?

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth +
              ROI:Condition:Depth,  # Effect of interest
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (4)
# Is there a difference in activation between conditions 'Kanizsa' and 'Kanizsa
# rotated'?

# Select 'Kanizsa' and 'Kanizsa roated' conditions:
objDfKonly <- objDf[objDf$Condition!='BRIGHT_SQUARE',]

# Null model:
mdlNull = lme(PSC ~ ROI + Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Depth,  # Effect of interest = Condition
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

#library(ggplot2)
#ggplot(data=objDf, aes(x=Depth, y=PSC, group=ROI, color=ROI)) +
#  geom_line()
