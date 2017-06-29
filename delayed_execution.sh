#!/bin/sh

###########################################################
### Bash utility to delay execution of a python script. ###
###########################################################

#-------------------------------------------------------------------------------
# Define parameters:

# Pause execution until (target epoch):
str_target_date="20170629 13:08:00"

# Array with paths of python scripts to call after target epoch is reached:
ary_paths=(/home/john/PhD/GitHub/py_depthsampling/delayed_execution_01_ds_crfMain_pow_uncor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_02_ds_crfMain_pow_cor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_03_ds_permMain_pow_uncor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_04_ds_permMain_pow_cor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_05_ds_crfMain_hyper_uncor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_06_ds_crfMain_hyper_cor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_07_ds_permMain_hyper_uncor.py \
           /home/john/PhD/GitHub/py_depthsampling/delayed_execution_08_ds_permMain_hyper_cor.py)

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
