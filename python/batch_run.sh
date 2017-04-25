#!/bin/bash

for pp in 0 0.00001 0.0001 0.001 0.01 0.1
do
	logdir="../log/pp_"${pp}
	mkdir -p ${logdir}
	for target_obj in Patient_1 Patient_2
	do
		python run.py $target_obj --win_size=20000 --corr_coef_pp=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
	for target_obj in Dog_1 Dog_2 Dog_3 Dog_4 Dog_5
	do
		python run.py $target_obj --corr_coef_pp=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done
