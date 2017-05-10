import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Dense, Flatten, Lambda
from keras.layers import Activation, BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import regularizers, optimizers
import tensorflow as tf
from sklearn import metrics
import util
import pandas as pd
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


DECAY_STEPS = 2000
DECAY_RATE = 0.95


class TsNet:

    def __init__(self, args, train_mean, train_std, num_channel):
        self.data_rebalance = args.data_rebalance
        self.reg_coef = args.reg_coef
        self.init_lr = args.init_lr
        self.win_size = args.win_size
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.corr_coef_pp = args.corr_coef_pp
        self.downsample = args.downsample
        self.test_data_dir = os.path.join(args.data_dir, args.target_obj)
        self.train_mean = train_mean
        self.train_std = train_std
        self.num_channel = num_channel
        if args.rand_scale_sampling:
            self.scales = [1, 2, 4]
        else:
            self.scales = [1,]
        logger.info("rand_scales={}".format(self.scales))

        self.weights_hist = []

    def build_model(self, logdir=None):

        def corr_loss_func(y_true, y_pred):
            return y_pred
        
        def gram_matrix(x):
            if K.ndim(x) == 2:
                features = K.transpose(x)
            else:
                features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
            gram = K.dot(features, K.transpose(features))
            return gram
        
        def corr_pp(x):
            pos_set1 = x[0]
            pos_set2 = x[1]
            pos_set1_avg = K.mean(pos_set1, axis=0)
            pos_set2_avg = K.mean(pos_set2, axis=0)
            gm1 = gram_matrix(pos_set1_avg)
            gm2 = gram_matrix(pos_set2_avg)
            return K.mean(K.square(gm1 - gm2))
        
        # outputs
        outputs = []
        losses = []
        loss_weights = []
        
        # inputs
        input_data = Input(shape=(self.win_size, self.num_channel),
                dtype="float32", name="data")
        pos_data1 = Input(shape=(self.win_size, self.num_channel),
                dtype="float32", name="pos_data1")
        pos_data2 = Input(shape=(self.win_size, self.num_channel),
                dtype="float32", name="pos_data2")
        
        # data normalization
        norm_layer = Lambda(lambda x: (x - self.train_mean) / self.train_std,
                name="normalization")
        x_all_data = norm_layer(input_data)
        x_pos_data1 = norm_layer(pos_data1)
        x_pos_data2 = norm_layer(pos_data2)
        
        conv_params = [(32, 8, 4), (64, 5, 2), (64, 2, 2)]
        n_convs = len(conv_params)
        loss_weights = [1.0 * self.corr_coef_pp / n_convs] * n_convs
        corr_layer = Lambda(corr_pp, name="corr_pp")
        for i, conv_param in enumerate(conv_params):
            num_filter, filter_size, pool_size = conv_param
            # conv
            conv_layer = Conv1D(num_filter, filter_size,
                    strides=pool_size,
                    kernel_regularizer=regularizers.l2(self.reg_coef),
                    name="conv" + str(i+1))
            x_all_data = conv_layer(x_all_data)
            x_pos_data1 = conv_layer(x_pos_data1)
            x_pos_data2 = conv_layer(x_pos_data2)
            # corr
            corr_pp = corr_layer([x_pos_data1, x_pos_data2])
            outputs.append(corr_pp)
            losses.append(corr_loss_func)
            # batch norm
            bn_layer = BatchNormalization()
            x_all_data = bn_layer(x_all_data)
            x_pos_data1 = bn_layer(x_pos_data1)
            x_pos_data2 = bn_layer(x_pos_data2)
            # relu
            activation_layer = Activation("relu")
            x_all_data = activation_layer(x_all_data)
            x_pos_data1 = activation_layer(x_pos_data1)
            x_pos_data2 = activation_layer(x_pos_data2)

        x_all_data = Flatten(name="flatten")(x_all_data)
        for i in xrange(2):
            # fc layers
            x_all_data = Dense(1024,
                    kernel_regularizer=regularizers.l2(self.reg_coef),
                    name="fc" + str(i+1))(x_all_data)
            # BN
            bn_layer = BatchNormalization()
            x_all_data = bn_layer(x_all_data)
            # relu
            activation_layer = Activation("relu")
            x_all_data = activation_layer(x_all_data)

        # output
        x_all_data = Dense(1, kernel_initializer="uniform")(x_all_data)
        x_all_data = BatchNormalization()(x_all_data)
        prob = Activation("sigmoid", name="prob")(x_all_data)
        
        outputs.append(prob)
        losses.append(binary_crossentropy)
        loss_weights.append(1.0)
        logger.info("loss_weights={}".format(loss_weights))
        
        self.num_outputs = len(outputs)
        logger.info(outputs)
        logger.info(loss_weights)
        self.model = Model(inputs=[input_data, pos_data1, pos_data2],
                outputs=outputs)
#        optimizer = optimizers.Adam(lr=self.init_lr, decay=1e-6)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(self.init_lr, self.global_step,
           DECAY_STEPS, DECAY_RATE, staircase=True)
        optimizer = optimizers.TFOptimizer(tf.train.AdamOptimizer(lr))

        self.model.compile(optimizer=optimizer,
                loss=losses, loss_weights=loss_weights)

        if logdir is not None:
            from keras.utils import plot_model
            plot_model(self.model,
                    to_file=os.path.join(logdir, 'model.png'),
                    show_shapes=True)

    def weights_squared_sum(self):
        weights_sum = 0
        for ll in self.model.layers:
            ll_name = ll.name
            if ll_name[:2] in ["co", "de", "fc"] and ll_name[:3]!="cor":
                ws = ll.get_weights()
                for w in ws:
                    weights_sum += np.sum(w**2)
        self.weights_hist.append(weights_sum)
        return weights_sum

    def get_rand_batch(self, data, labels):
        data_size, seq_len, num_ch = data.shape
        pos_label_idx = np.where(labels==1)[0]
        logger.info("pos_label_idx={}".format(pos_label_idx))
        num_pos_samples = np.sum(labels==1)
        num_neg_samples = np.sum(labels==0)
        pos_sampling_weight = 1.0
        if self.data_rebalance:
            pos_sampling_weight *= (1.0 * num_neg_samples / num_pos_samples)
        logger.info("pos_sampling_weight={}".format(
            pos_sampling_weight))
        sampling_weights = np.ones(data_size)
        sampling_weights[labels==1] *= pos_sampling_weight
        sampling_weights /= np.sum(sampling_weights)

        logger.info("data_fetcher off to work")
        while True:
            sample_idx = np.random.choice(data_size, self.batch_size,
                    p=sampling_weights)

            # prevent batches of only negative samples
            if np.intersect1d(sample_idx, pos_label_idx).size == 0:
                sample_idx[0] = pos_label_idx[np.random.randint(0,
                    pos_label_idx.size)]

            batch_label = labels[sample_idx]
            assert np.any(batch_label==1)

            sample_idx = np.expand_dims(sample_idx, -1)
            pos_idx1 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)
            pos_idx2 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)

            scale_list = np.random.choice(self.scales, self.batch_size)
            logger.debug("scale_list={}".format(scale_list))
            batch_data = np.zeros([self.batch_size, self.win_size,
                self.num_channel])
            batch_pos_data1 = np.zeros_like(batch_data)
            batch_pos_data2 = np.zeros_like(batch_data)
            uniq_scales = np.unique(scale_list)
            sample_weights = np.ones(self.batch_size)
            for w, scale in enumerate(uniq_scales):
                sel_ix = np.where(scale_list==scale)[0]
                sample_weights[sel_ix] = w+1
                rand_start = np.random.randint(0,
                        seq_len - self.win_size * scale + 1, sel_ix.size)
                seq_slice = np.array([np.arange(ss,
                    ss + self.win_size * scale, scale)
                    for ss in rand_start])
                batch_data[sel_ix] = data[sample_idx[sel_ix], seq_slice]
                batch_pos_data1[sel_ix] = data[pos_idx1[sel_ix], seq_slice]
                batch_pos_data2[sel_ix] = data[pos_idx2[sel_ix], seq_slice]
            logger.debug("sample_weights={}".format(sample_weights))
            yield ([batch_data, batch_pos_data1, batch_pos_data2],
                    [batch_label]*self.num_outputs,
                    [sample_weights]*self.num_outputs)
        logger.info("data_fetcher abort") 

#    def get_ordered_batch(self, data, labels):
#        data_size, seq_len, _ = data.shape
#        while True:
#            for i in xrange(data_size):
#                batch_data = []
#                batch_label = []
#                sample_weights = []
#                for w, scale in enumerate(self.scales):
#                    tmp = util.reshape(data[i], self.win_size, scale)
#                    batch_data.append(tmp)
#                    batch_label.append(np.ones(tmp.shape[0]) * labels[i])
#                    sample_weights.extend([w+1]*tmp.shape[0])
#                batch_data = np.concatenate(batch_data)
#                batch_label = np.concatenate(batch_label)
#                sample_weights = np.asarray(sample_weights)
#                yield ([batch_data]*3, [batch_label]*self.num_outputs,
#                        [sample_weights]*self.num_outputs) 

    def train(self, train_data, train_label, valid_data, valid_label,
            logdir, bestmodelpath, finalmodelpath, verbose=0):

        steps_per_epoch = 1000
        validation_steps = valid_data.shape[0]
        logger.info("max_epochs={}, steps_per_epoch={}, "\
                "validation_steps={}".format(
                    self.max_epochs, steps_per_epoch, validation_steps))

        self.best_acc = 0.0
        self.best_f1 = 0.0

        def test(dd_test, ll_test):
            test_size = ll_test.size
            probs = np.zeros(test_size * len(self.scales))
            for i in xrange(ll_test.size):
                for j, scale in enumerate(self.scales):
                    batch_data = util.reshape(dd_test[i], self.win_size, scale)
                    preds_per_ts = self.model.predict_on_batch([batch_data] * 3)
                    probs[j * test_size + i] = preds_per_ts[-1].mean()
            probs = np.mean(np.reshape(probs, (len(self.scales), test_size)),
                    axis=0)
            preds = np.array(probs>=0.5, dtype=int)
            acc = metrics.accuracy_score(ll_test, preds)
            f1 = metrics.f1_score(ll_test, preds)
            prec = metrics.precision_score(ll_test, preds)
            rec = metrics.recall_score(ll_test, preds)
            return acc, f1, prec, rec

        def test_on_train(epoch, logs):
            rand_ix = np.random.choice(train_label.size,
                    valid_label.size, replace=False)
            acc, f1, prec, rec = test(train_data[rand_ix], train_label[rand_ix])
            logger.info("------------")
            logger.info("Epoch={}, entropy_loss={}, "\
                    " weights_squared_sum={}".format(
                        epoch, logs["prob_loss"], self.weights_squared_sum()))
            logger.info("Epoch={}, train_acc={}, "\
                    "train_f1={}, train_prec={}, train_recall={}".format(
                        epoch, acc, f1, prec, rec))

        def test_on_valid(epoch, logs):
            acc, f1, prec, rec = test(valid_data, valid_label)
            logger.info("------------")
            logger.info("Epoch={}, val_acc={}, "\
                    "val_f1={}, val_prec={}, val_recall={}".format(
                        epoch, acc, f1, prec, rec))
            # update best model
            if epoch>=(self.max_epochs/3) and acc>self.best_acc or \
                    (acc==self.best_acc and f1>=self.best_f1):
                self.best_acc = acc
                self.best_f1 = f1
                logger.info("Best scores updated!")
                logger.info("Updating best model ...")
                self.model.save_weights(bestmodelpath)
                logger.info("Model saved.")

        train_hist = self.model.fit_generator(
            generator=self.get_rand_batch(train_data, train_label),
            steps_per_epoch=steps_per_epoch,
#            validation_data=self.get_ordered_batch(valid_data, valid_label),
#            validation_steps=validation_steps,
            callbacks=[
                keras.callbacks.LambdaCallback(
                    on_epoch_end=test_on_train)
                ,
                keras.callbacks.LambdaCallback(
                    on_epoch_end=test_on_valid)
                ],
            verbose=verbose,
            epochs=self.max_epochs
            )

        logger.info("Saving final model ...")
        self.model.save_weights(finalmodelpath)
        logger.info("Model saved.")

        train_hist.history["weights_squared_sum"] = self.weights_hist
        return train_hist

    def test_on_data(self, test_data_files):
        """Test model performance on the test set"""
        test_size = test_data_files.size
        probs = np.zeros(test_size * len(self.scales))
        for i, f in enumerate(test_data_files):
            test_data = util.load_data(
                    os.path.join(self.test_data_dir, f),
                    "test", self.downsample)
            for j, scale in enumerate(self.scales):
                batch_data = util.reshape(test_data,
                        self.win_size, scale)
                preds_per_ts = self.model.predict_on_batch([batch_data] * 3)
                probs[j * test_size + i] = preds_per_ts[-1].mean()
        probs = np.mean(np.reshape(probs, (len(self.scales), test_size)),
                axis=0)
        return probs
