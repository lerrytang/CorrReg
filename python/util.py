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
    # create model dir for this run
    modeldir = os.path.join(logdir, "model")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    return logdir, modeldir


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
    return data, data_length_sec, sampling_frequency, channels, sequence


def load_data_for_test(filename, win_size):
    data, _, _, _, _ = load_data(filename, "test")
    # channel last
    data = np.transpose(data, axes=[1, 0])
    return reshape(data, win_size)


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
    for idx, datafile in enumerate(datafiles):
        filename = os.path.join(dirname, datafile)
        res = load_data(filename, datatype)
        data, data_length_sec, sampling_frequency, channels, sequence = res
        if all_data is None:
            num_channel, seq_len = data.shape
            all_data = np.zeros([len(datafiles), seq_len, num_channel],
                    dtype=data.dtype)
        all_data[idx] = np.transpose(data, axes=[1, 0])
    logger.info("Data from {} (shape:{}, dtype:{})".format(dirname,
        all_data.shape, all_data.dtype))
    return all_data


def load_train_data(target_data_dir):
    # load data
    data_files = os.listdir(target_data_dir)
    pre_data_files = np.sort([f for f in data_files if "preictal" in f])
    logger.info("#preictal_files = {}".format(pre_data_files.size))
    int_data_files = np.sort([f for f in data_files if "interictal" in f])
    logger.info("#interictal_files = {}".format(int_data_files.size))
    pre_data = load_all_data(target_data_dir, "preictal")
    logger.info("interictal_data.shape={}".format(pre_data.shape))
    int_data = load_all_data(target_data_dir, "interictal")
    logger.info("interictal_data.shape={}".format(int_data.shape))
    # split
    labels = np.array([1] * pre_data_files.size + \
            [0] * int_data_files.size, dtype="uint8")
    data = np.concatenate([pre_data, int_data])
    return data, labels


def split_to_folds(labels, n_folds=2):
    train_sets = []
    valid_sets = []
    skf = StratifiedKFold(labels, n_folds, shuffle=True)
    for train_ix, valid_ix in skf:
        train_sets.append(train_ix)
        valid_sets.append(valid_ix)
    return train_sets, valid_sets


def reshape(data, win_size):
    # reshape data
    seq_len = data.shape[0]
    segs_per_ts = int(np.ceil(seq_len / win_size))
    stride = int((seq_len - win_size) / (segs_per_ts - 1))
    logger.debug("seq_len={}, segs_per_ts={}, stride={}".format(
        seq_len, segs_per_ts, stride))
    start_idx = np.arange(0, seq_len - win_size + 1,
            stride, dtype="int32")
    assert start_idx.size == segs_per_ts
    slice_idx = np.array([np.arange(ss, ss + win_size) for ss in start_idx])
    return data[slice_idx]