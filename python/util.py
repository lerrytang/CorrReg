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
    return logdir, modeldir


def load_data(filename, datatype, downsample=0):
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
    seq_len, num_ch = data.shape
    if downsample > 1:
        data = reshape(data, downsample)
        logger.debug("data.shape={}".format(data.shape))
        downsample_mean = np.mean(data, axis=1)
        logger.debug("downsample_mean.shape={}".format(downsample_mean.shape))
        downsample_std = np.std(data, axis=1)
        data = np.concatenate([downsample_mean, downsample_std], axis=-1)
    return data


def load_all_data(dirname, datatype, downsample=0):
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
        data = load_data(filename, datatype, downsample)
        if all_data is None:
            seq_len, num_channel = data.shape
            all_data = np.zeros([len(datafiles), seq_len, num_channel],
                    dtype=data.dtype)
        all_data[idx] = np.copy(data)
    logger.info("Data from {} (shape:{}, dtype:{})".format(dirname,
        all_data.shape, all_data.dtype))
    return all_data


def load_train_data(target_data_dir, downsample):
    data_files = os.listdir(target_data_dir)
    pre_data_files = np.sort([f for f in data_files if "preictal" in f])
    logger.info("#preictal_files = {}".format(pre_data_files.size))
    int_data_files = np.sort([f for f in data_files if "interictal" in f])
    logger.info("#interictal_files = {}".format(int_data_files.size))
    pre_data = load_all_data(target_data_dir, "preictal", downsample)
    logger.info("interictal_data.shape={}".format(pre_data.shape))
    int_data = load_all_data(target_data_dir, "interictal", downsample)
    logger.info("interictal_data.shape={}".format(int_data.shape))
    labels = np.array([1] * pre_data_files.size + \
            [0] * int_data_files.size, dtype="uint8")
    data = np.concatenate([pre_data, int_data])
    logger.info("data.dtype={}".format(data.dtype))
    return data, labels


def split_to_folds(labels, n_folds=2):
    train_sets = []
    valid_sets = []
    skf = StratifiedKFold(labels, n_folds, shuffle=True)
    for train_ix, valid_ix in skf:
        train_sets.append(train_ix)
        valid_sets.append(valid_ix)
    return train_sets, valid_sets


def reshape(data, win_size, skip=1):
    assert np.ndim(data)==2
    seq_len, num_ch = data.shape
    segs_per_ts = int(np.ceil(1.0 * seq_len / win_size * skip))
    logger.debug("seq_len={}, num_ch={}, segs_per_ts={}".format(
        seq_len, num_ch, segs_per_ts))
    stride = int((seq_len - win_size * skip) / (segs_per_ts - 1))
    start_idx = np.arange(0, seq_len - win_size * skip + 1, stride)
    slice_idx = np.array([np.arange(ss, ss + win_size * skip, skip)
        for ss in start_idx])
    reshaped_data = data[slice_idx]
    return reshaped_data


