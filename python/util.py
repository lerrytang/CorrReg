import numpy as np
import pandas as pd
import scipy.io
from sklearn.cross_validation import StratifiedKFold
import time
import os
import sys
import logging
logger = logging.getLogger(__name__)


def create_log(log_dir, target_obj):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    current_time = time.strftime("%Y%m%d%H%M%S")
    logdir = os.path.join(log_dir, target_obj + "_" + current_time)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    modeldir = os.path.join(logdir, "model")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    return logdir


def load_data(filename, datatype):
    '''
    Utility for loading .mat files
    '''
    # sanity check
    assert datatype in ["preictal", "interictal", "test"],\
            "datatype should be ('preictal', 'interictal', 'test')"
    if not os.path.exists(filename):
        raise ValueError("Specified file ({}) does not exist.".format(filename))
    #extract data from .mat file
    logger.info("filename={}".format(filename))
    raw_data = scipy.io.loadmat(filename, chars_as_strings=True)
    seq_idx = int(os.path.basename(filename).split(".")[0].split("_")[-1])
    field_name = datatype + "_segment_" + str(seq_idx)
    matdata= raw_data[field_name].flatten()
    # load each field
    data = matdata[0][0]
    data_length_sec = matdata[0][1].flatten()[0]
    sampling_frequency = np.round(matdata[0][2].flatten()[0])
    channels = matdata[0][3].flatten()
    channels = np.array([ch[0] for ch in channels])
    if datatype == "test":
        sequence = 1
    else:
        sequence = matdata[0][4].flatten()[0]
    # ensure channel order
    data = data[np.argsort(channels)]
    # channel last
    data = np.transpose(data, axes=[1, 0])
    return data, sequence


def load_all_data(dirname, datatype):
    '''
    Load all data of datatype from a directory
    '''
    # sanity check
    if not os.path.exists(dirname):
        raise ValueError("Directory ({}) does not exist.".format(dirname))
    logger.info("Loading {} data from {} ...".format(datatype, dirname))
    # load data
    datafiles = os.listdir(dirname)
    datafiles = np.sort([f for f in datafiles if datatype in f])
    all_data = None
    ss_ind = []  # start index of independent samples
    ee_ind = []  # end index of independent samples
    prev_seq_id = 1
    for idx, datafile in enumerate(datafiles):
        filename = os.path.join(dirname, datafile)
        data, seq_id = load_data(filename, datatype)
        if all_data is None:
            seq_len, num_channel = data.shape
            all_data = np.zeros([len(datafiles), seq_len, num_channel],
                    dtype=data.dtype)
        all_data[idx] = np.copy(data)
        if seq_id <= prev_seq_id:
            ss_ind.append(idx)
            if idx-1 > 0:
                ee_ind.append(idx-1)
        prev_seq_id = seq_id
    else:
        ee_ind.append(idx)
    assert len(ss_ind)==len(ee_ind), "{} vs {}".format(len(ss_ind), len(ee_ind))
    logger.info("Data from {} (shape:{}, dtype:{})".format(dirname,
        all_data.shape, all_data.dtype))
    return all_data, np.asarray(ss_ind), np.asarray(ee_ind)


def load_train_data(target_data_dir):
    data_files = os.listdir(target_data_dir)
    pre_data_files = np.sort([f for f in data_files if "preictal" in f])
    logger.info("#preictal_files = {}".format(pre_data_files.size))
    int_data_files = np.sort([f for f in data_files if "interictal" in f])
    logger.info("#interictal_files = {}".format(int_data_files.size))
    pre_data, ss_pre_data, ee_pre_data =\
            load_all_data(target_data_dir, "preictal")
#    logger.info("interictal_data.shape={}".format(pre_data.shape))
#    logger.info("start of independent samples: {}".format(ss_pre_data))
    logger.info("end of independent samples: {}".format(ee_pre_data))
    int_data, ss_int_data, ee_int_data =\
            load_all_data(target_data_dir, "interictal")
    ss_int_data += pre_data.shape[0]
    ee_int_data += pre_data.shape[0]
    logger.info("interictal_data.shape={}".format(int_data.shape))
#    logger.info("start of independent samples: {}".format(ss_int_data))
#    logger.info("end of independent samples: {}".format(ee_int_data))
    labels = np.array([1] * pre_data_files.size + \
            [0] * int_data_files.size, dtype="uint8")
    data = np.concatenate([pre_data, int_data])
    logger.info("data.dtype={}".format(data.dtype))
    return data, labels, (ss_pre_data, ee_pre_data), (ss_int_data, ee_int_data)


def split_to_folds(labels, n_folds=3):
    train_sets = []
    valid_sets = []
    skf = StratifiedKFold(labels, n_folds, shuffle=True)
    for train_ix, valid_ix in skf:
        train_sets.append(train_ix)
        valid_sets.append(valid_ix)
    return train_sets, valid_sets

#def split_to_folds(pos_ix, neg_ix, n_folds=2):
#    ss_pos, ee_pos = pos_ix
#    logger.info("ss_pos={}".format(ss_pos))
#    logger.info("ee_pos={}".format(ee_pos))
#    ss_neg, ee_neg = neg_ix
#    labels = [1]*ss_pos.size + [0]*ss_neg.size
#    ss_ix = np.concatenate([ss_pos, ss_neg])
#    ee_ix = np.concatenate([ee_pos, ee_neg])
#    assert len(labels)==ss_ix.size==ee_ix.size
#
#    train_sets = []
#    valid_sets = []
#    skf = StratifiedKFold(labels, n_folds, shuffle=True)
#    for train_ix, valid_ix in skf:
#        train_ss = ss_ix[train_ix]
#        train_ee = ee_ix[train_ix]
#        expanded_train_ix = []
#        for i in range(train_ss.size):
#            expanded_train_ix.extend(np.arange(train_ss[i], train_ee[i]+1).tolist())
#        expanded_train_ix = np.array(expanded_train_ix)
##        logger.info("expanded_train_ix={}".format(expanded_train_ix))
#
#        valid_ss = ss_ix[valid_ix]
#        valid_ee = ee_ix[valid_ix]
#        expanded_valid_ix = []
#        for i in range(valid_ss.size):
#            expanded_valid_ix.extend(np.arange(valid_ss[i], valid_ee[i]+1).tolist())
#        expanded_valid_ix = np.array(expanded_valid_ix)
##        logger.info("expanded_valid_ix={}".format(expanded_valid_ix))
#
#        train_sets.append(expanded_train_ix)
#        valid_sets.append(expanded_valid_ix)
#
#    return train_sets, valid_sets


def reshape(data, win_size, skip=1):
    assert np.ndim(data)==2
    seq_len, num_ch = data.shape
    segs_per_ts = int(np.ceil(1.0 * seq_len / (win_size * skip)))
    logger.debug("skip={}, seq_len={}, num_ch={}, segs_per_ts={}".format(
        skip, seq_len, num_ch, segs_per_ts))
    stride = int((seq_len - win_size * skip) / (segs_per_ts - 1))
    start_idx = np.arange(0, seq_len - win_size * skip + 1, stride)
    slice_idx = np.array([np.arange(ss, ss + win_size * skip, skip)
        for ss in start_idx])
    reshaped_data = data[slice_idx]
    return reshaped_data

