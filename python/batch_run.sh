#!/bin/bash

#data_dir="$HOME/ts_prediction_keras/data/"
for pp in 0 0.00001 0.0001 0.001 0.01 0.1
do
	logdir="../log/pp_"${pp}
	mkdir -p ${logdir}
	for target_obj in Patient_2 Dog_1 Dog_2 Dog_3 Dog_4 Dog_5 Patient_1
	do
#		python run.py $target_obj --corr_coef_pp=${pp} --data_dir=${data_dir} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
		python run.py $target_obj --corr_coef_pp=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done
