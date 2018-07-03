
library(ggplot2)

strCsv <- '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf/dataframe.csv'

dfSlope = read.csv(strCsv)

colnames(dfSlope)

ggplot(data=dfSlope, mapping=aes(x=ROI, y=Slope, group=Depth,)) + geom_boxplot()
