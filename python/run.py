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

    if not args.test:
        logdir, modeldir = util.create_log(args.logdir, args.target_obj)
    else:
        logdir = args.logdir

    # load data for training
    target_data_dir = os.path.join(args.data_dir, args.target_obj)
    logger.info("target_data_dir={}".format(target_data_dir))
    data, labels = util.load_train_data(target_data_dir, args.downsample)
    train_ix, valid_ix = util.split_to_folds(labels, args.n_folds)
    split_file = os.path.join(logdir, "train_valid_split.pkl")
    with open(split_file, "wb") as f:
        pickle.dump((train_ix, valid_ix), f)
    num_seq, seq_len, num_ch = data.shape
    logger.info("num_seq={}, seq_len={}, num_ch={}".format(
        num_seq, seq_len, num_ch))

    # CV
    for fold_i in xrange(args.n_folds):
        logger.info("<fold {}>".format(fold_i))
        logger.info("---------------")

        # split data
        train_indice = train_ix[fold_i]
        valid_indice = valid_ix[fold_i]
        logger.info(valid_indice)
        logger.info("pos% in train: {}".format(labels[train_indice].mean()))
        logger.info("pos% in valid: {}".format(labels[valid_indice].mean()))
    
        # calculate mean and std
        npzdir = os.path.join(args.train_mean_std_dir, args.target_obj)
        filename = "train_mean_std_fold{}.npz".format(fold_i)
        if os.path.exists(os.path.join(npzdir, filename)):
            logger.info("Loading {} ...".format(filename))
            npzfile = np.load(os.path.join(npzdir, filename))
            train_mean = npzfile["train_mean"]
            train_std = npzfile["train_std"]
        else:
            logger.info("Calculating mean and std ...")
            tmp = data[train_indice].reshape([-1, num_ch])
            train_mean = np.mean(tmp, axis=0)
            train_std = np.std(tmp, axis=0)
        logger.info("train_mean.shape={}".format(train_mean.shape))
        logger.info("train_mean={}".format(train_mean))
        logger.info("train_std.shape={}".format(train_std.shape))
        logger.info("train_std={}".format(train_std))
        np.savez(os.path.join(logdir, filename),
                train_mean=train_mean,
                train_std=train_std)

        # build net
        logger.info("Build model")
        model = TsNet(args, train_mean, train_std, num_ch)
        model.build_model(logdir=logdir if fold_i==0 else None)
    
        modeldir = os.path.join(logdir, "model")
        bestmodelpath = os.path.join(modeldir,
                "bestmodel_fold" + str(fold_i) + ".h5")
        finalmodelpath = os.path.join(modeldir,
                "finalmodel_fold" + str(fold_i) + ".h5")
        if args.rand_scale_sampling:
            bestdiripath = os.path.join(modeldir,
                    "bestdiri_fold" + str(fold_i) + ".npz")
            finaldiripath = os.path.join(modeldir,
                    "finaldiri_fold" + str(fold_i) + ".npz")

        # train
        if not args.test:
            train_hist = model.train(data[train_indice], labels[train_indice],
                    data[valid_indice], labels[valid_indice],
                    logdir, bestmodelpath, finalmodelpath, args.verbose)
            # log training history
            hist_file = os.path.join(logdir, "hist_fold"+str(fold_i)+".pkl")
            with open(hist_file, "wb") as f:
                pickle.dump(train_hist.history, f)
    
        # load test data
        if args.use_final_model:
            logger.info("Load final model to test")
            model.model.load_weights(finalmodelpath)
            if args.rand_scale_sampling:
                dirichlet_file = np.load(finaldiripath)
                model.model.dirichlet = dirichlet_file["dirichlet"]
            logger.info("model.model.dirichlet={}".format(
                model.model.dirichlet))
        else:
            logger.info("Load best model to test")
            model.model.load_weights(bestmodelpath)
            if args.rand_scale_sampling:
                dirichlet_file = np.load(bestdiripath)
                model.model.dirichlet = dirichlet_file["dirichlet"]
            logger.info("model.model.dirichlet={}".format(
                model.model.dirichlet))
           
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
    parser.add_argument("--test", action="store_true", default=False,
            help="whether to test")
    parser.add_argument("--use_final_model", action="store_true", default=False,
            help="whether to test with the model trained until max_epochs")
    parser.add_argument("--data_rebalance", action="store_true", default=False,
            help="whether to rebalance dataset")
    parser.add_argument("--rand_scale_sampling", action="store_true", default=False,
            help="whether to random sampling with different scales")
    parser.add_argument("--data_dir", default="../data",
            help="directory of data")
    parser.add_argument("--train_mean_std_dir", default="../train_mean_std_dir",
            help="directory of mean and std files")
    parser.add_argument("--logdir", default="./log",
            help="directory to store logs")
    parser.add_argument("--corr_coef_pp", default=0.0, type=float,
            help="coefficient for triplet loss (positive and positive")
    parser.add_argument("--downsample", default=0, type=int,
            help="ratio to downsample data")
    parser.add_argument("--win_size", default=4000, type=int,
            help="size of sliding window")
    parser.add_argument("--batch_size", default=64, type=int,
            help="training batch size")
    parser.add_argument("--max_epochs", default=50, type=int,
            help="maximum number of training iterations")
    parser.add_argument("--n_folds", default=3, type=int,
            help="training batch size")
    parser.add_argument("--rand_seed", default=11, type=int,
            help="random seed for reproducibility")
    parser.add_argument("--verbose", default=0, type=int,
            help="verbose for training process")
    parser.add_argument("--reg_coef", default=0.0, type=float,
            help="L2 regularization strength")
    parser.add_argument("--init_lr", default=0.001, type=float,
            help="initial learning rate")
    parser.add_argument("target_obj",
            help="must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")
    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    main(args)
