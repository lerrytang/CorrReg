#!/bin/bash

for pp in 0
do
	logdir="../log/baseline"
	mkdir -p ${logdir}
	python run.py Dog_1 --corr_coef_pp=${pp} --data_rebalance --n_folds=4 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_1.txt
	python run.py Dog_2 --corr_coef_pp=${pp} --data_rebalance --n_folds=7 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_2.txt
	python run.py Dog_3 --corr_coef_pp=${pp} --data_rebalance --n_folds=12 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_3.txt
	python run.py Dog_4 --corr_coef_pp=${pp} --data_rebalance --n_folds=17 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_4.txt
	python run.py Dog_5 --corr_coef_pp=${pp} --data_rebalance --n_folds=5 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_5.txt
	for target_obj in Patient_1 Patient_2
	do
		python run.py $target_obj --downsample=10 --corr_coef_pp=${pp} --data_rebalance --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done
