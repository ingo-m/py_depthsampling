#!/bin/sh

###########################################################
### Bash utility to delay execution of a python script. ###
###########################################################

#-------------------------------------------------------------------------------
# Define parameters:

# Pause execution until (target epoch):
str_target_date="20170628 20:05:55"

# Array with paths of python scripts to call after target epoch is reached:
ary_paths=(/home/john/Desktop/pyecho01.py \
           /home/john/Desktop/pyecho02.py)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Delay

echo "------------Delayed execution: Sleep until ${str_target_date}-------------"

# Convert target date into epoch:
target_epoch=$(date -d "${str_target_date}" +%s)

# Get current time in seconds (seconds from epoch):
current_epoch=$(date +%s)

# How many seconds to sleep until target epoch is reached:
sleep_seconds=$(( $target_epoch - $current_epoch ))

# Sleep:
sleep $sleep_seconds
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Execute scripts

echo "------------Delayed execution: Awake.-------------"

for index_1 in ${ary_paths[@]}
do
	echo "------------Executing: ${index_1}"

	python "${index_1}"

done
#-------------------------------------------------------------------------------
