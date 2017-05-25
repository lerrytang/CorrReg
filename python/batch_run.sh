#!/bin/bash

# baseline + corr_reg
for pp in 0 0.00001 0.0001 0.001 0.01 0.1
do
	logdir="../log/baseline_"${pp}
	mkdir -p ${logdir}
	python run.py Dog_1 --n_folds=4 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_1.txt
	python run.py Dog_2 --n_folds=7 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_2.txt
	python run.py Dog_3 --n_folds=12 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_3.txt
	python run.py Dog_4 --n_folds=17 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_4.txt
	python run.py Dog_5 --n_folds=5 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_5.txt
	for target_obj in Patient_1 Patient_2
	do
		python run.py $target_obj --downsample=10 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done


# multi-scale + corr_reg
for pp in 0 0.00001 0.0001 0.001 0.01 0.1
do
	logdir="../log/multiscale_"${pp}
	mkdir -p ${logdir}
	python run.py Dog_1 --n_folds=4 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_1.txt
	python run.py Dog_2 --n_folds=7 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_2.txt
	python run.py Dog_3 --n_folds=12 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_3.txt
	python run.py Dog_4 --n_folds=17 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_4.txt
	python run.py Dog_5 --n_folds=5 --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_5.txt
	for target_obj in Patient_1 Patient_2
	do
		python run.py $target_obj --downsample=10 --multiscale --corr_reg=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
	done
done
