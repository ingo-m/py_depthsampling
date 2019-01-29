
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data.csv', header=TRUE, sep=';')

head(objDf)

# # Linear model using generalised least squares:
# mdl_gls = gls(PSC ~ ROI * Condition * Depth * Subject,
#               objDf,
#               correlation=corCAR1(form=(~1|Depth)),
#               method='ML')
# 
# summary(mdl_gls)
# 
# nlme:::summary.gls(mdl_gls)$tTable

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
              Condition:Depth:ROI,  # Effect of interest
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# Model comparison:
anova(mdlNull, mdlFull)

#library(ggplot2)
#ggplot(data=objDf, aes(x=Depth, y=PSC, group=ROI, color=ROI)) +
#  geom_line()


