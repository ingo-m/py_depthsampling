
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/depth_data.csv', header=TRUE, sep=';')

head(objDf)

# lme(PSC~ROI+Condition+Depth, objDf, random=~Depth|Subject)

# objLme = lme(PSC ~ ROI + Condition + Depth,
#              objDf,
#              random=(~1|Subject),
#              method='ML',
#              correlation=corCAR1(form=(~1|Depth)))

# Linear mixed-effects model:
mdl_lme = lme(PSC ~ ROI * Condition * Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

# # Linear model using generalised least squares:
# mdl_gls = gls(PSC ~ ROI * Condition * Depth * Subject,
#               objDf,
#               correlation=corCAR1(form=(~1|Depth)),
#               method='ML')
# 
# summary(mdl_gls)
# 
# nlme:::summary.gls(mdl_gls)$tTable

mdlNull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

mdlFull = lme(PSC ~ ROI + Condition + Depth +
              ROI:Condition +
              ROI:Depth +
              Condition:Depth +
              Condition:Depth:ROI,
              objDf,
              random=(~1|Subject),
              correlation=corCAR1(form=(~1|Subject/Depth)),
              method='ML')

anova(mdlNull, mdlFull)

#library(ggplot2)
#ggplot(data=objDf, aes(x=Depth, y=PSC, group=ROI, color=ROI)) +
#  geom_line()


