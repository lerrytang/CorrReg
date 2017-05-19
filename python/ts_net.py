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

NUM_SCALES = 4
T = 0.2

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
        self.multiscale = args.multiscale
        if self.multiscale:
            self.scales = np.arange(NUM_SCALES) + 1
            self.theta = np.ones(NUM_SCALES) 
        else:
            self.scales = np.array([1,])
            self.theta = np.array([1.0])
        logger.info("self.scales={}".format(self.scales))
        logger.info("self.theta={}".format(self.theta))

        self.weights_hist = []

    @property
    def scale_weights(self):
        tmp = np.exp(self.theta / T)
        return tmp / tmp.sum()

    def build_model(self, logdir=None):

        def corr_loss_func(y_true, y_pred):
            return K.mean(y_pred * y_true)
        
        def corr_pp(x):
            pos_set1 = x[0]
            pos_set2 = x[1]
            # gram matrix
            gm1 = tf.matmul(pos_set1, pos_set1, transpose_a=True)
            gm2 = tf.matmul(pos_set2, pos_set2, transpose_a=True)
            logger.info("gm1.shape={}".format(gm1.shape))
            logger.info("gm2.shape={}".format(gm2.shape))
            gm_mse = K.mean(K.batch_flatten(tf.square(gm1-gm2)),
                    axis=0, keepdims=True)
            logger.info("gm_mse.shape={}".format(gm_mse.shape))
            return gm_mse
      
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
            # corr
            corr_pp = corr_layer([x_pos_data1, x_pos_data2])
            outputs.append(corr_pp)
            losses.append(corr_loss_func)

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
        optimizer = optimizers.Adam(lr=self.init_lr, decay=1e-6)
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
        logger.info("#pos_samples={}".format(pos_label_idx.size))
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

        logger.info("get_rand_batch off to work")
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

            # sample scales
            scale_list = np.random.choice(self.scales,
                    self.batch_size,
                    p=self.scale_weights if self.multiscale else None)
            batch_data = np.zeros([self.batch_size, self.win_size,
                self.num_channel], dtype=float)
            batch_pos_data1 = np.zeros_like(batch_data)
            batch_pos_data2 = np.zeros_like(batch_data)
            batch_w = np.zeros_like(batch_label, dtype=float)
            uniq_scales = np.unique(scale_list)
            for w, scale in enumerate(uniq_scales):
                sel_ix = np.where(scale_list==scale)[0]
                rand_start = np.random.randint(0,
                        seq_len - self.win_size * scale + 1, sel_ix.size)
                seq_slice = np.array([np.arange(ss,
                    ss + self.win_size * scale, scale)
                    for ss in rand_start])
                batch_data[sel_ix] = data[sample_idx[sel_ix], seq_slice]
                batch_pos_data1[sel_ix] = data[pos_idx1[sel_ix], seq_slice]
                batch_pos_data2[sel_ix] = data[pos_idx2[sel_ix], seq_slice]
                batch_w[sel_ix] = 1.0 / self.batch_size
            assert np.allclose(batch_w.sum(), 1.0), "batch_w.sum()={}".format(batch_w.sum())
            yield ([batch_data, batch_pos_data1, batch_pos_data2],
                    [batch_w]*(self.num_outputs-1) + [batch_label])

    def get_batch(self, data, labels):
        data_size = labels.size
        logger.info("get_batch off to work (data_size={})".format(data_size))
        while True:
            for s, scale in enumerate(self.scales):
                for i in xrange(data_size):
                    batch_data = util.reshape(data[i], self.win_size, scale)
                    batch_label = np.ones(batch_data.shape[0]) * labels[i]
                    batch_weights = \
                            np.ones(batch_data.shape[0]) * self.scale_weights[s]
                    rand_ix = np.arange(batch_data.shape[0])
                    np.random.shuffle(rand_ix)
                    yield ([batch_data]*2+[batch_data[rand_ix]],
                            [batch_label]*self.num_outputs,
                            [batch_weights]*self.num_outputs)

    def train(self, train_data, train_label, valid_data, valid_label,
            logdir, bestmodelpath, finalmodelpath,
            best_theta_path=None, final_theta_path=None, verbose=0):

        steps_per_epoch = 200
        validation_steps = valid_data.shape[0]
        logger.info("max_epochs={}, steps_per_epoch={}, "\
                "validation_steps={}".format(
                    self.max_epochs, steps_per_epoch, validation_steps))

        self.min_val_prob_loss = np.inf

        def update_theta(epoch, logs):

            # log weights by the way
            self.weights_squared_sum()

            # print stats
            logger.info("Epoch={}, train_loss={}, train_prob_loss={}, "\
                    "val_prob_loss={}, L2(weights)={}".format(
                        epoch+1, logs["loss"],logs["prob_loss"],
                        logs["val_prob_loss"], self.weights_hist[-1]))
            if self.corr_coef_pp>0:
                logger.info("\tcorr_pp_loss={}".format(
                    [logs["corr_pp_loss_" + str(i+1)]
                        for i in xrange(self.num_outputs-1)]))

            # sample train data for testing
            if self.multiscale:
                rand_ix = np.random.choice(train_label.size,
                        valid_label.size, replace=False)
                ll_test = train_label[rand_ix]
                dd_test = train_data[rand_ix]
                logger.info("Updating theta ...")
                logger.info("\tBefore update: theta={}, scale_weights={}".format(
                    self.theta, self.scale_weights))
                log_likelihood = np.zeros_like(self.theta)
                for s, scale in enumerate(self.scales):
                    probs = np.zeros(ll_test.size)
                    corr_loss = 0.0
                    for i in xrange(ll_test.size):
                        batch_data = util.reshape(dd_test[i], self.win_size, scale)
                        preds_per_ts = self.model.predict_on_batch([batch_data] * 3)
                        probs[i] = preds_per_ts[-1].mean()
                        for k in xrange(self.num_outputs-1):
                            corr_loss += preds_per_ts[k].mean()
                    neg_mask = ll_test==0
                    probs[neg_mask] = 1-probs[neg_mask]
                    log_likelihood[s] = np.mean(np.log(probs)) - self.corr_coef_pp * corr_loss
                exp_theta = np.exp(self.theta)
                y_s = exp_theta / exp_theta.sum()
                self.theta += log_likelihood * y_s * (1-y_s) / T
                logger.info("\tAfter update: theta={}, scale_weights={}".format(
                    self.theta, self.scale_weights))

            # save model if necessary
            val_prob_loss = logs["val_prob_loss"]
            if val_prob_loss <= self.min_val_prob_loss:
                self.min_val_prob_loss = val_prob_loss
                self.model.save_weights(bestmodelpath)
                if self.multiscale:
                    np.savez(best_theta_path, theta=self.theta)
                logger.info("Best model updated.")

            logger.info("-"*50)

        train_hist = self.model.fit_generator(
            generator=self.get_rand_batch(train_data, train_label),
            steps_per_epoch=steps_per_epoch,
            validation_data=self.get_batch(valid_data, valid_label),
            validation_steps=self.scales.size*valid_label.size,
            callbacks=[
                keras.callbacks.LambdaCallback(
                    on_epoch_end=update_theta),
                keras.callbacks.EarlyStopping(
                    monitor="val_prob_loss",
                    min_delta=0.0001,
                    patience=10,
                    mode="min"),
#                keras.callbacks.TensorBoard(
#                    log_dir=logdir,
#                    histogram_freq=2,
#                    write_images=True),
                ],
            verbose=verbose,
            epochs=self.max_epochs
            )

        logger.info("Saving final model ...")
        self.model.save_weights(finalmodelpath)
        if final_theta_path is not None:
            logger.info("Saving theta to {}".format(final_theta_path))
            np.savez(final_theta_path, theta=self.theta)
        logger.info("Model saved.")

        train_hist.history["weights_squared_sum"] = self.weights_hist
        return train_hist

    def test_on_data(self, test_data_files):
        """Test model performance on the test set"""
        test_size = test_data_files.size
        prob_matrix = np.zeros([test_size, self.scales.size], dtype=float)
        logger.info("self.scale_weights={}".format(self.scale_weights))
        for i, f in enumerate(test_data_files):
            test_data, _ = util.load_data(
                    os.path.join(self.test_data_dir, f),
                    "test", self.downsample)
            for j, scale in enumerate(self.scales):
                batch_data = util.reshape(test_data,
                        self.win_size, scale)
                preds_per_ts = self.model.predict_on_batch([batch_data] * 3)
                prob_matrix[i, j] = preds_per_ts[-1].mean()
        probs = prob_matrix.dot(self.scale_weights)
        return probs
