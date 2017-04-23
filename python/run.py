import pickle
import util
import ts_net
import argparse
import time
import pandas as pd
import numpy as np
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
    res.to_csv(os.path.join(logdir, "csv_to_submit.csv"))


def main(args):
    logdir, modeldir = util.create_log(args.logdir, args.target_obj)
    # load data for training
    target_data_dir = os.path.join(args.data_dir, args.target_obj)
    logger.info("target_data_dir={}".format(target_data_dir))
    data, labels = util.load_train_data(target_data_dir)
    train_sets, valid_sets = util.split_to_folds(data, labels, args.n_folds)
    # CV
    for fold_i in xrange(args.n_fold):
        logger.info("<fold {}>".format(fold_i))
        logger.info("---------------")

        # calculate mean and std
        train_data, train_labels = train_sets[fold_i]
        valid_data, valid_labels = valid_sets[fold_i]
        num_channel = train_data.shape[-1]
        tmp = train_data.reshape([-1, num_channel])
        train_mean = np.mean(tmp, axis=0)
        train_std = np.std(tmp, axis=0)
        
        # build net
        logger.info("Build model")
        model = TsNet(args, train_mean, train_std,
                train_data, train_labels, valid_data, valid_labels)
        model.build_model()
        model.build_func()

        # train
        logger.info("Start to train")
        modelname = "bestmodel_fold" + str(fold_i) + ".h5"
        modelpath = os.path.join(logdir, "model", modelname)
        train_hist = model.train(logdir, modelpath)

        # log training history
        hist_file = os.path.join(logdir, "hist_fold"+str(fold_i)+".pkl")
        with open(hist_file, "wb") as f:
            pickle.dump(train_hist.history, f)
            
        # load test data
        del train_data, train_labels, valid_data, valid_labels
        data_files = os.listdir(target_data_dir)
        test_data_files = np.sort([f for f in data_files if "test" in f])
        logger.info("#test_files = {}".format(test_data_files.size))
        test_data = util.load_all_data(target_data_dir, "test")
        logger.info("test_data.shape={}".format(test_data.shape))

        # test
        logger.info("Load model to test")
        model.model.load_weights(os.path.join(logdir, "model", modelname))
        preds = model.test_on_data(test_data)
        output = pd.Series(preds, index=test_data_files)
        output_file = os.path.join(logdir, "output_fold"+str(fold_i)+".csv")
        output.to_csv(output_file)
        logger.info("Test result written to {}.".format(output_file))

    merge_results(logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./log",
            help="directory to store logs")
    parser.add_argument("--corr_coef_pp", default=0.0, type=float,
            help="coefficient for triplet loss (positive and positive")
    parser.add_argument("--win_size", default=8000, type=int,
            help="size of sliding window")
    parser.add_argument("--batch_size", default=32, type=int,
            help="training batch size")
    parser.add_argument("--max_epochs", default=20, type=int,
            help="maximum number of training iterations")
    parser.add_argument("--n_folds", default=3, type=int,
            help="training batch size")
    parser.add_argument("--verbose", default=0, type=int,
            help="verbose for training process")
    parser.add_argument("--reg_coef", default=0.0, type=float,
            help="L2 regularization strength")
    parser.add_argument("--dropout_prob", default=0.0, type=float,
            help="dropout probability")
    parser.add_argument("target_obj",
            help="must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")
    args = parser.parse_args()

    main(args)
