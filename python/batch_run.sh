#!/bin/bash

# baseline
logdir="../log/baseline"
mkdir -p ${logdir}
python run.py Dog_1 --corr_coef_pp=${pp} --n_folds=4 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_1.txt
python run.py Dog_2 --corr_coef_pp=${pp} --n_folds=7 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_2.txt
python run.py Dog_3 --corr_coef_pp=${pp} --n_folds=12 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_3.txt
python run.py Dog_4 --corr_coef_pp=${pp} --n_folds=17 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_4.txt
python run.py Dog_5 --corr_coef_pp=${pp} --n_folds=5 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_5.txt
for target_obj in Patient_1 Patient_2
do
	python run.py $target_obj --downsample=10 --corr_coef_pp=${pp} --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
done


# multi-scale + cor_reg
for pp in 0.000001 0.00001 0.0001 0.001 0.01 0.1
do
	logdir="../log/multiscale_pp_"${pp}
	mkdir -p ${logdir}
	python run.py Dog_1 --corr_coef_pp=${pp} --multiscale --n_folds=4 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_1.txt
	python run.py Dog_2 --corr_coef_pp=${pp} --multiscale --n_folds=7 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_2.txt
	python run.py Dog_3 --corr_coef_pp=${pp} --multiscale --n_folds=12 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_3.txt
#	python run.py Dog_4 --corr_coef_pp=${pp} --multiscale --n_folds=17 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_4.txt
#	python run.py Dog_5 --corr_coef_pp=${pp} --multiscale --n_folds=5 --logdir=${logdir} 2>&1 | tee ${logdir}/log_Dog_5.txt
#	for target_obj in Patient_1 Patient_2
#	do
#		python run.py $target_obj --downsample=10 --corr_coef_pp=${pp} --multiscale --logdir=${logdir} 2>&1 | tee ${logdir}/log_${target_obj}.txt
#	done
done
