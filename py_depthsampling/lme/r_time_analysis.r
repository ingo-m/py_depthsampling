
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/centre/ert_lme.csv', header=TRUE, sep=';')
#objDf <- read.csv(file='/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/edge/ert_lme.csv', header=TRUE, sep=';')
#objDf <- read.csv(file='/home/john/Dropbox/Surface_Depth_Data/Higher_Level_Analysis/background/ert_lme.csv', header=TRUE, sep=';')

head(objDf)

# Weights = ~1/n where n is number of vertices.

# (1) - Simple effect ROI

# Null model:
mdlNull = lme(PSC ~ Condition + Volume,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: ROI
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (2) - Simple effect condition

# Null model:
mdlNull = lme(PSC ~ ROI + Volume,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: condition
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (3) - Simple effect time

# Null model:
mdlNull = lme(PSC ~ ROI + Condition,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: time
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (4) - Interaction time by condition

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Volume +
              ROI:Condition +
              ROI:Volume,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume +
              ROI:Condition +
              ROI:Volume +
              Condition:Volume,  # Effect of interest
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# Select 'Kanizsa' and 'Kanizsa roated' conditions:
objDfKonly <- objDf[objDf$Condition!='BRIGHT_SQUARE',]

# (5) - Simple effect ROI

# Null model:
mdlNull = lme(PSC ~ Condition + Volume,
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: ROI
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (6) - Simple effect condition

# Null model:
mdlNull = lme(PSC ~ ROI + Volume,
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: condition
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (7) - Simple effect time

# Null model:
mdlNull = lme(PSC ~ ROI + Condition,
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume,  # Effect of interest: time
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

# (8) - Interaction time by condition

# Null model:
mdlNull = lme(PSC ~ ROI + Condition + Volume +
              ROI:Condition +
              ROI:Volume,
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Full model:
mdlFull = lme(PSC ~ ROI + Condition + Volume +
              ROI:Condition +
              ROI:Volume +
              Condition:Volume,  # Effect of interest
              objDfKonly,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Volume)),
              weights=(~1/Vertices),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

#library(ggplot2)
#objDfTmp <- objDfKonly[objDfKonly$ROI=='V1',]
#objDfTmp <- aggregate(PSC ~ Volume/Condition, objDfKonly, mean)
#ggplot(data=objDfTmp, aes(x=Volume, y=PSC, group=Condition, color=Condition)) +
#  geom_line() +
#  geom_point()
