#!/bin/bash

for pp in 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000
do
	logdir="../log/rand_scale_pp_"${pp}
	mkdir -p ${logdir}
	for target_obj in Dog_1 Dog_2 Dog_3 Dog_4 Dog_5
	do
		python run.py $target_obj --corr_coef_pp=${pp} --data_rebalance --rand_crop_sampling --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
	for target_obj in Patient_1 Patient_2
	do
		python run.py $target_obj --downsample=10 --corr_coef_pp=${pp} --data_rebalance --rand_crop_sampling --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done
