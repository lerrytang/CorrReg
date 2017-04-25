import pickle
import util
from ts_net import TsNet
import argparse
import time
import pandas as pd
import numpy as np
import gc
import os
import logging
logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def merge_results(logdir):
    output_files = [f for f in os.listdir(logdir) if f[:6]=="output"]
    res = None
    for f in output_files:
        tmp = pd.read_csv(os.path.join(logdir, f), index_col=0, header=None)
        if res is None:
            res = tmp
        else:
            res = res + tmp
    res /= len(output_files)
    res.to_csv(os.path.join(logdir, "csv_to_submit.csv"), header=False)
    return res


def main(args):
    for arg in vars(args):
        logger.info("{} = {}".format(arg, getattr(args, arg)))
    logdir, modeldir = util.create_log(args.logdir, args.target_obj)

    # load data for training
    target_data_dir = os.path.join(args.data_dir, args.target_obj)
    logger.info("target_data_dir={}".format(target_data_dir))
    data, labels = util.load_train_data(target_data_dir)
    train_ix, valid_ix = util.split_to_folds(labels, args.n_folds)
    split_file = os.path.join(logdir, "train_valid_split.pkl")
    with open(split_file, "wb") as f:
        pickle.dump((train_ix, valid_ix), f)
    num_seq, seq_len, num_ch = data.shape
    logger.info("num_seq={}, seq_len={}, num_ch={}".format(
        num_seq, seq_len, num_ch))

    # load mean and std
    npzfile_path = os.path.join(args.train_mean_std_dir,
            args.target_obj, "train_mean_std.npz")
    logger.info("npzfile_path={}".format(npzfile_path))
    npzfile = np.load(npzfile_path)
    train_mean = npzfile["train_mean"]
    train_std = npzfile["train_std"]

    # CV
    for fold_i in xrange(args.n_folds):
        logger.info("<fold {}>".format(fold_i))
        logger.info("---------------")

        # split data
        train_indice = train_ix[fold_i]
        valid_indice = valid_ix[fold_i]
    
        # build net
        logger.info("Build model")
        model = TsNet(args, train_mean, train_std, num_ch)
        model.build_model()
    
        # train
        modelname = "bestmodel_fold" + str(fold_i) + ".h5"
        modelpath = os.path.join(logdir, "model", modelname)
        train_hist = model.train(data[train_indice], labels[train_indice],
                data[valid_indice], labels[valid_indice],
                logdir, modelpath, args.verbose)
    
        # log training history
        hist_file = os.path.join(logdir, "hist_fold"+str(fold_i)+".pkl")
        with open(hist_file, "wb") as f:
            pickle.dump(train_hist.history, f)
    
        # load test data
        logger.info("Load model to test")
        model.model.load_weights(os.path.join(logdir, "model", modelname))
           
        # test
        data_files = os.listdir(target_data_dir)
        test_data_files = np.sort([f for f in data_files if "test" in f])
        logger.info("#test_files = {}".format(test_data_files.size))
        preds = model.test_on_data(test_data_files)
        output = pd.Series(preds, index=test_data_files)
        output_file = os.path.join(logdir, "output_fold"+str(fold_i)+".csv")
        output.to_csv(output_file)
        logger.info("Test result written to {}.".format(output_file))

    merge_results(logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data",
            help="directory of data")
    parser.add_argument("--logdir", default="./log",
            help="directory to store logs")
    parser.add_argument("--train_mean_std_dir", default="../train_mean_std_dir",
            help="directory of train_mean_std.npz")
    parser.add_argument("--corr_coef_pp", default=0.0, type=float,
            help="coefficient for triplet loss (positive and positive")
    parser.add_argument("--win_size", default=4000, type=int,
            help="size of sliding window")
    parser.add_argument("--batch_size", default=32, type=int,
            help="training batch size")
    parser.add_argument("--max_epochs", default=50, type=int,
            help="maximum number of training iterations")
    parser.add_argument("--n_folds", default=3, type=int,
            help="training batch size")
    parser.add_argument("--verbose", default=0, type=int,
            help="verbose for training process")
    parser.add_argument("--reg_coef", default=0.0, type=float,
            help="L2 regularization strength")
    parser.add_argument("--dropout_prob", default=0.0, type=float,
            help="dropout probability")
    parser.add_argument("--init_lr", default=0.0002, type=float,
            help="initial learning rate")
    parser.add_argument("target_obj",
            help="must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")
    args = parser.parse_args()

    main(args)
