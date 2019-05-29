# GLS

# install.packages('nlme')
library(nlme)

# Read CSV into R
objDf <- read.csv(file='/home/john/PhD/GitLab/py_depthsampling/py_depthsampling/lme/anovarm.csv', header=TRUE, sep=';')

#objGls = gls(PSC~ROI+Condition+Depth, objDf, correlation=corLin(form=~Depth+ROI+Condition|Subject))
#objGls = gls(PSC~ROI*Condition*Depth, objDf, correlation=corLin(form=~1|Depth))
#summary(objGls)
#nlme:::summary.gls(objGls)$tTable

# Linear model using generalised least squares:
mdl_gls = gls(PSC ~ ROI + Condition + Depth,
              objDf,
              method='ML',
              correlation=corCAR1(form=(~1|Depth)))

mdl_gls2 = gls(PSC ~ ROI + Condition + Depth +
               Depth:ROI,
               objDf,
               method='ML',
               correlation=corCAR1(form=(~1|Depth)))

anova(mdl_gls, mdl_gls2)
