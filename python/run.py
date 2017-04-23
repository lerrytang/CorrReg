import data_util
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


def test(data_dir, model_dir, args):
    # sanity check
    npzfile_path = os.path.join(model_dir, MEAN_STD_FILE)
    if not os.path.exists(npzfile_path):
        raise ValueError("train_mean_std.npz not found in ({}).".format(model_dir))

    # load mean and std from training data
    npzfile = np.load(npzfile_path)
    train_mean = npzfile["train_mean"]
    train_std = npzfile["train_std"]

    # load test data
    logger.info("Target: {}".format(args.target_obj))
    target_data_dir = os.path.join(data_dir, args.target_obj)
    data_files = os.listdir(target_data_dir)
    test_data_files = np.sort([f for f in data_files if "test" in f])
    logger.info("#test_files = {}".format(test_data_files.size))
    test_data = data_util.load_all_data(target_data_dir, "test")
    logger.info("test_data.shape={}".format(test_data.shape))
    num_channel = test_data.shape[-1]
    assert num_channel == train_mean.size,\
            "#channel do not match ({} vs {})".format(num_channel, train_mean.size)

    with tf.Session() as sess:
        model = ts_net.TsPredNet(args.batch_size, num_channel, train_mean, train_std,
                args.win_size, args.reg_coef)
        corr_coef = (args.corr_coef_pn, args.corr_coef_pp, args.corr_coef_nn)
        model.build_net(corr_coef)
        init = tf.global_variables_initializer()
        sess.run(init)

        # load model
        if args.best_valid:
            subdir = "best_valid_model"
        else:
            subdir = "model"
        res = model.load_net(sess, os.path.join(model_dir, subdir))
        if not res:
            logger.error("Trained model not loaded, abort")
            return

        # test on test set
        res, _, _, _, _ = model.test_on_data(sess, test_data)
        output = pd.Series(res, index=test_data_files)
        output_file = os.path.join(model_dir, args.output_filename)
        output.to_csv(output_file)
        logger.info("Test result written to {}.".format(output_file))


def train(data_dir, log_dir, args):
    # create log dir for this run
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    current_time = time.strftime("%Y%m%d%H%M%S")
    logdir = os.path.join(log_dir, args.target_obj + "_" + current_time)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    # create model dir for this run
    modeldir = os.path.join(logdir, "model")
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    bv_modeldir = os.path.join(logdir, "best_valid_model")
    if not os.path.exists(bv_modeldir):
        os.mkdir(bv_modeldir)

    # log to file as well
    fh = logging.FileHandler(os.path.join(logdir, "train_log.txt"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    for arg in vars(args):
        logger.info("{} = {}".format(arg, getattr(args, arg)))

    # load data
    logger.info("Target: {}".format(args.target_obj))
    target_data_dir = os.path.join(data_dir, args.target_obj)
    data_files = os.listdir(target_data_dir)
    preictal_data_files = np.sort([f for f in data_files if "preictal" in f])
    logger.info("#preictal_files = {}".format(preictal_data_files.size))
    interictal_data_files = np.sort([f for f in data_files if "interictal" in f])
    logger.info("#interictal_files = {}".format(interictal_data_files.size))
    preictal_data = data_util.load_all_data(target_data_dir, "preictal")
    logger.info("interictal_data.shape={}".format(preictal_data.shape))
    interictal_data = data_util.load_all_data(target_data_dir, "interictal")
    logger.info("interictal_data.shape={}".format(interictal_data.shape))

    # split data into train and valid
    train_ratio = args.train_ratio 
    res = data_util.split_train_valid(train_ratio, preictal_data, interictal_data)
    train_data, train_labels, valid_data, valid_labels = res
    logger.debug("train_data.shape={}".format(train_data.shape))
    logger.debug("train_labels.shape={}".format(train_labels.shape))
    if valid_data is not None:
        logger.debug("valid_data.shape={}".format(valid_data.shape))
        logger.debug("valid_labels.shape={}".format(valid_labels.shape))

    # get statistics for data normalization
    num_channel = train_data.shape[-1]
    npzfile_path = os.path.join(args.train_mean_std_dir, MEAN_STD_FILE)
    if os.path.exists(npzfile_path):
        # load mean and std from training data
        npzfile = np.load(npzfile_path)
        train_mean = npzfile["train_mean"]
        train_std = npzfile["train_std"]
    else:
        # compute training mean and std (memory aggresive)
        tmp = train_data.reshape([-1, num_channel])#.astype("int16")
        train_mean = np.mean(tmp, axis=0)
        train_std = np.std(tmp, axis=0)
    logger.debug("train_mean.shape: {}".format(train_mean.shape))
    logger.debug("train_mean: {}".format(train_mean))
    logger.debug("train_std.shape: {}".format(train_std.shape))
    logger.debug("train_std: {}".format(train_std))
    np.savez(os.path.join(logdir, MEAN_STD_FILE),
            train_mean=train_mean, train_std=train_std)
#    np.savez(os.path.join(logdir, "valid_set.npz"),
#            valid_data=valid_data, valid_labels=valid_labels)

    # training
    corr_coef = (args.corr_coef_pn, args.corr_coef_pp, args.corr_coef_nn)
    with tf.device("/gpu:0"):
        model = ts_net.TsPredNet(args.batch_size, num_channel,
                train_mean, train_std, args.win_size, args.reg_coef,
                args.dropout_prob, args.init_lr)
        model.build_net(corr_coef)
        init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(init)
        best_iter = model.train(logdir, modeldir, bv_modeldir,
                train_data, train_labels, valid_data, valid_labels,
                log_n_iter=args.log_n_iter, test_n_iter=args.test_n_iter,
                max_iter=args.max_iter)
    return logdir, best_iter


def main(args):

    root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(root_dir, "data")
    logger.info("data_dir={}".format(data_dir))

    if args.test:
        logger.info("Prepare for testing ...")
        if not os.path.exists(args.model_dir):
            raise ValueError("Specified model dir ({}) does not exist.".format(args.model_dir))
        test(data_dir, args.model_dir, args)
    else:
        logger.info("Prepare for training ...")
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
        logdir, best_iter = train(data_dir, args.logdir, args)

        # load the lastly saved model and test
        logger.info("Prepare for testing with the lastly saved model ...")
        tf.reset_default_graph() 
        test(data_dir, logdir, args)

        # load the model with best validation accuracy and test
        if best_iter > 0:
            args.best_valid = True
            args.output_filename = "output_best_valid.csv"
            logger.info("Prepare for testing with the model of best validation accuracy ...")
            tf.reset_default_graph() 
            test(data_dir, logdir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true", default=False,
            help="whether to test a trained model")
    parser.add_argument("--model_dir", default="./model",
            help="directory of trained model")
    parser.add_argument("--logdir", default="./log",
            help="directory to store logs")
    parser.add_argument("--best_valid", action="store_true", default=False,
            help="to restore a checkpoint from a certain iteration")
    parser.add_argument("--output_filename", default="output.csv",
            help="filename of the output file")
    parser.add_argument("--rand_seed", default=11, type=int,
            help="random seed for numpy.random")

    # triplet loss related
    parser.add_argument("--corr_coef_pn", default=0.0, type=float,
            help="coefficient for triplet loss (positive and negative")
    parser.add_argument("--corr_coef_nn", default=0.0, type=float,
            help="coefficient for triplet loss (negative and negative")
    parser.add_argument("--corr_coef_pp", default=0.0, type=float,
            help="coefficient for triplet loss (positive and positive")

    # experimental design related
    parser.add_argument("--train_ratio", default=0.7, type=float,
            help="training data ratio")
    parser.add_argument("--win_size", default=4000, type=int,
            help="size of sliding window")
    parser.add_argument("--batch_size", default=64, type=int,
            help="training batch size")
    parser.add_argument("--log_n_iter", default=200, type=int,
            help="log training status every n iterations")
    parser.add_argument("--test_n_iter", default=200, type=int,
            help="perform test on training/validation set every n iterations")

    # training related
    parser.add_argument("--train_mean_std_dir", default="./data",
            help="directory of pre-computed mean and std of training data")
    parser.add_argument("--max_iter", default=50000, type=int,
            help="maximum number of training iterations")
    parser.add_argument("--init_lr", default=0.001, type=float,
            help="initial learning rate")
    parser.add_argument("--reg_coef", default=0.0, type=float,
            help="L2 regularization strength")
    parser.add_argument("--dropout_prob", default=0.0, type=float,
            help="dropout probability")

    # mandatory
    parser.add_argument("target_obj",
            help="must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")
    
    args = parser.parse_args()
    if args.target_obj not in ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]:
        raise ValueError("target_obj must be in the set (Dog_1, Dog_2, Dog_3,"
            " Dog_4, Dog_5, Patient_1, Patient_2)")

    assert args.batch_size % 4 == 0, "Set batch size to be mutliple of 4"

    # for reproducability
    RAND_SEED = args.rand_seed
    np.random.seed(RAND_SEED)

    main(args)

