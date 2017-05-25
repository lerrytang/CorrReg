#!/bin/bash

for pp in 0 0.00001 0.0001 0.001 0.01 0.1
do
	for multiscale in "" "--multiscale"
	do
		logdir="../log/pp_"${pp}${multiscale}
		mkdir -p ${logdir}
		for target_obj in Dog_1 Dog_2 Dog_3 Dog_4 Dog_5
		do
			python run.py $target_obj ${multiscale} --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
		done
		for target_obj in Patient_1 Patient_2
		do
			python run.py $target_obj ${multiscale} --downsample=10 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
		done
	done
done
