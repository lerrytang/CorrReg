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


def get_mean_std(data, args):
    npzdir = os.path.join(args.train_mean_std_dir, args.target_obj)
    npzfile = os.path.join(npzdir, "train_mean_std.npz")
    if os.path.exists(npzfile):
        logger.info("Loading from {} ...".format(npzfile))
        npzdata = np.load(npzfile)
        all_mean = npzdata["all_mean"]
        all_std = npzdata["all_std"]
    else:
        logger.info("Calculating mean and std ...")
        all_mean = np.mean(data, axis=1)
        all_std = np.std(data, axis=1)
    logger.info("all_mean.shape={}".format(all_mean.shape))
    logger.info("all_mean={}".format(all_mean))
    logger.info("all_std.shape={}".format(all_std.shape))
    logger.info("all_std={}".format(all_std))
    if not os.path.exists(npzdir):
        os.makedirs(npzdir)
    if not os.path.exists(npzfile):
        np.savez(npzfile, all_mean=all_mean, all_std=all_std)
    return all_mean, all_std


def main(args):

    # log parameters for the trial
    for arg in vars(args):
        logger.info("{} = {}".format(arg, getattr(args, arg)))

    # define folders
    if not args.test:
        logdir = util.create_log(args.logdir, args.target_obj)
    else:
        logdir = args.logdir
    modeldir = os.path.join(logdir, "model")

    # load data
    target_data_dir = os.path.join(args.data_dir, args.target_obj)
    logger.info("target_data_dir={}".format(target_data_dir))
    data, labels, pos_ix, neg_ix =\
            util.load_train_data(target_data_dir)
    train_ix, valid_ix = util.split_to_folds(pos_ix, neg_ix, args.n_folds)
    num_seq, seq_len, num_ch = data.shape
    logger.info("num_seq={}, seq_len={}, num_ch={}".format(
        num_seq, seq_len, num_ch))

    # calculate mean and std
    all_mean, all_std = get_mean_std(data, args)

    # CV
    for fold_id in xrange(args.n_folds):
        logger.info("fold: {}".format(fold_id))

        # split data
        train_indice = train_ix[fold_id]
        valid_indice = valid_ix[fold_id]
        logger.info("train_ix={}".format(train_indice))
        logger.info("valid_ix={}".format(valid_indice))
        logger.info("train: #pos={}, #neg={}, %pos={}".format(
            labels[train_indice].sum(),
            train_indice.size - labels[train_indice].sum(),
            labels[train_indice].mean()))
        logger.info("valid: #pos={}, #neg={}, %pos={}".format(
            labels[valid_indice].sum(),
            valid_indice.size - labels[valid_indice].sum(),
            labels[valid_indice].mean()))

        # build net
        logger.info("Build model")
        train_mean = all_mean[train_indice].mean(axis=0)
        train_std = all_std[train_indice].mean(axis=0)
        model = TsNet(args, train_mean, train_std, num_ch)
        model.build_model(logdir=logdir if fold_id==0 else None)
    
        modelpath = os.path.join(modeldir, "model_{}.h5".format(fold_id))
        if args.multiscale:
            thetapath = os.path.join(modeldir, "theta_{}.npz".format(fold_id))
        else:
            thetapath = None
    
        # train
        if not args.test:
            train_hist = model.train(
                    data[train_indice], labels[train_indice],
                    data[valid_indice], labels[valid_indice],
                    logdir, modelpath, thetapath, args.verbose)
            hist_file = os.path.join(logdir, "hist_{}.pkl".format(fold_id))
            with open(hist_file, "wb") as f:
                pickle.dump(train_hist.history, f)

        # test
        data_files = os.listdir(target_data_dir)
        test_data_files = np.sort([f for f in data_files if "test" in f])
        logger.info("#test_files = {}".format(test_data_files.size))

        logger.info("modelpath={}".format(modelpath))
        model.model.load_weights(modelpath)
        if args.multiscale:
            logger.info("thetapath={}".format(thetapath))
            theta_file = np.load(thetapath)
            model.model.theta = theta_file["theta"]
            logger.info("model.model.theta={}".format(model.model.theta))
        preds = model.test_on_data(test_data_files)
        output = pd.Series(preds, index=test_data_files)
        output_file = os.path.join(logdir,
                "{}_preds{}.csv".format(args.target_obj, fold_id))
        output.to_csv(output_file)
        logger.info("Test result written to {}.".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", default=False,
            help="whether to test")
    parser.add_argument("--multiscale", action="store_true", default=False,
            help="whether to apply multiscale sampling")
    parser.add_argument("--data_dir", default="../data",
            help="directory of data")
    parser.add_argument("--train_mean_std_dir", default="../train_mean_std_dir",
            help="directory of mean and std files")
    parser.add_argument("--logdir", default="./log",
            help="directory to store logs")
    parser.add_argument("--corr_reg", default=0.0, type=float,
            help="coefficient for CorrReg")
    parser.add_argument("--n_folds", default=3, type=int,
            help="number of CV folds")
    parser.add_argument("--downsample", default=0, type=int,
            help="ratio to downsample data")
    parser.add_argument("--win_size", default=4000, type=int,
            help="size of sliding window")
    parser.add_argument("--batch_size", default=256, type=int,
            help="training batch size")
    parser.add_argument("--max_epochs", default=150, type=int,
            help="maximum number of training epoches")
    parser.add_argument("--rand_seed", default=11, type=int,
            help="random seed for reproducibility")
    parser.add_argument("--verbose", default=0, type=int,
            help="verbose for training process")
    parser.add_argument("--reg_coef", default=0.00001, type=float,
            help="L2 regularization strength")
    parser.add_argument("--init_lr", default=0.001, type=float,
            help="initial learning rate")
    parser.add_argument("target_obj",
            help="must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")
    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    main(args)
