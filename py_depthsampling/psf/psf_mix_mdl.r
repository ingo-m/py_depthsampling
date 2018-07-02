

# Install library for reading json files:
# install.packages("rjson")

json_file <- '/Users/john/Dropbox/PacMan_Depth_Data/Higher_Level_Analysis/psf/dataframe.json'

#library("rjson")
#json_data <- fromJSON(file=json_file)

# install.packages("jsonlite")
library('jsonlite')

json_data <- fromJSON(json_file)

colnames(json_data)
