import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Dense, Flatten, Lambda, Activation
#from keras.layers import BatchNormalization
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
        self.reg_coef = args.reg_coef
        self.downsample = args.downsample
        self.init_lr = args.init_lr
        self.win_size = args.win_size
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.corr_reg = args.corr_reg
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
        if self.downsample > 0:
            self.scales *= self.downsample
        logger.info("self.scales={}".format(self.scales))
        logger.info("self.theta={}".format(self.theta))

        self.weights_hist = []

    @property
    def scale_weights(self):
        tmp = np.exp(self.theta / T)
        return tmp / tmp.sum()

    def build_model(self, logdir=None):

        def corr_loss_func(y_true, y_pred):
            """
            y_pred - flatten Gram matix of shape Bx(CxC)
                     where B for batch_size, C for #filter
            y_true - maks matrix of shape BxP,
                     where P is #positives in the batch
                     it looks like
                     | 0 0 0 1 0 0 |
                     | 0 0 0 0 1 0 |
                     | 0 0 0 0 0 1 |
                     if B=6 and only the last 3 samples in the batch are positve
            """
            pos_gram = K.dot(K.transpose(y_true), y_pred)  #  Px(CxC)
            pos_std = K.var(pos_gram, axis=0)  # var accross the batch
            return K.mean(pos_std)

        def corr_pp(x):
            """
            x is of shape BxLxC, where B for batch_size,
            L for seq_len, C for #filters
            """
            gm = tf.matmul(x, x, transpose_a=True)
            logger.info("gm.shape={}".format(gm.shape))
            gm_flat = K.batch_flatten(gm)
            return gm_flat
     
        # outputs
        outputs = []
        losses = []
        loss_weights = []
        
        # inputs
        input_data = Input(shape=(self.win_size, self.num_channel),
                dtype="float32", name="data")
       
        # data normalization
        norm_layer = Lambda(lambda x: (x - self.train_mean) / self.train_std,
                name="normalization")
        x_all_data = norm_layer(input_data)
        
        conv_params = [(32, 8, 4), (64, 5, 2), (64, 2, 2)]
        n_convs = len(conv_params)
        loss_weights = [self.corr_reg]
        corr_layer = Lambda(corr_pp, name="corr_pp")
        for i, conv_param in enumerate(conv_params):
            num_filter, filter_size, pool_size = conv_param
            # conv
            conv_layer = Conv1D(num_filter, filter_size,
                    strides=pool_size,
                    kernel_regularizer=regularizers.l2(self.reg_coef),
                    name="conv" + str(i+1))
            x_all_data = conv_layer(x_all_data)
            # relu
            activation_layer = Activation("relu")
            x_all_data = activation_layer(x_all_data)

        # corr
        corr_pp = corr_layer(x_all_data)
        outputs.append(corr_pp)
        losses.append(corr_loss_func)

        x_all_data = Flatten(name="flatten")(x_all_data)
        for i in xrange(2):
            # fc layers
            x_all_data = Dense(1024,
                    kernel_regularizer=regularizers.l2(self.reg_coef),
                    name="fc" + str(i+1))(x_all_data)
            # relu
            activation_layer = Activation("relu")
            x_all_data = activation_layer(x_all_data)

        # output
        prob = Dense(1, activation="sigmoid", name="prob")(x_all_data)
        
        outputs.append(prob)
        losses.append(binary_crossentropy)
        loss_weights.append(1.0)
        logger.info("loss_weights={}".format(loss_weights))
        
        self.num_outputs = len(outputs)
        logger.info(outputs)
        logger.info(loss_weights)
        self.model = Model(inputs=[input_data], outputs=outputs)

#        optimizer = optimizers.Adam(lr=self.init_lr, decay=1e-5)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(self.init_lr, global_step,
                DECAY_STEPS, DECAY_RATE, staircase=True)
        optimizer = keras.optimizers.TFOptimizer(tf.train.AdamOptimizer(lr))

        self.model.compile(optimizer=optimizer,
                loss=losses, loss_weights=loss_weights, metrics=["acc"])

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
        pos_sampling_weight = (1.0 * num_neg_samples / num_pos_samples)
        logger.info("pos_sampling_weight={}".format(pos_sampling_weight))
        sampling_weights = np.ones(data_size)
        sampling_weights[labels==1] *= pos_sampling_weight
        sampling_weights /= np.sum(sampling_weights)

        logger.info("get_rand_batch off to work")
        while True:
            sample_idx = np.random.choice(data_size, self.batch_size,
                    p=sampling_weights)

            batch_label = labels[sample_idx]
            assert np.any(batch_label==1)

            sample_idx = np.expand_dims(sample_idx, -1)

            # sample scales
            scale_list = np.random.choice(self.scales, self.batch_size,
                    p=self.scale_weights if self.multiscale else None)
            batch_data = np.zeros([self.batch_size, self.win_size,
                self.num_channel], dtype=float)
            uniq_scales = np.unique(scale_list)
            for scale in uniq_scales:
                sel_ix = np.where(scale_list==scale)[0]
                rand_start = np.random.randint(0,
                        seq_len - self.win_size * scale + 1, sel_ix.size)
                seq_slice = np.array([np.arange(ss,
                    ss + self.win_size * scale, scale)
                    for ss in rand_start])
                batch_data[sel_ix] = data[sample_idx[sel_ix], seq_slice]
            batch_mask = np.zeros([self.batch_size, (batch_label==1).size])
            batch_mask[:, np.where(batch_label==1)[0]] = 1
            batch_mask = [batch_mask] * (self.num_outputs - 1)
            yield([batch_data], batch_mask + [batch_label])

    def get_batch(self, data, labels):
        data_size = labels.size
        logger.info("get_batch off to work (data_size={})".format(data_size))
        while True:
            for s, scale in enumerate(self.scales):
                for i in xrange(data_size):
                    batch_data = util.reshape(data[i], self.win_size, scale)
                    bsize = batch_data.shape[0]
                    batch_label = np.ones(bsize) * labels[i]
                    batch_weights = np.ones(bsize) * self.scale_weights[s]
                    batch_weights = [batch_weights] * self.num_outputs
                    if labels[i]==1:
                        batch_mask = np.eye(batch_data.shape[0])
                    else:
                        batch_mask = np.zeros([bsize, 2])
                        # select only the first sample such that var=0
                        batch_mask[:, 0] = 1
                    batch_mask = [batch_mask] * (self.num_outputs - 1)
                    if self.multiscale:
                        yield([batch_data], batch_mask+[batch_label], batch_weights)
                    else:
                        yield([batch_data], batch_mask+[batch_label])

    def train(self, train_data, train_label, logdir,
            modelpath=None, thetapath=None, verbose=0):

        steps_per_epoch = 200
        logger.info("max_epochs={}, steps_per_epoch={}".format(
                    self.max_epochs, steps_per_epoch))

        def update_theta(epoch, logs):

            # log weights by the way
            self.weights_squared_sum()

            # print stats
            logger.info("Epoch={}, acc={}, loss={}, prob_loss={}, "\
                    "corr_loss={}, L2(weights)={}".format(
                        epoch+1, logs["prob_acc"], logs["loss"], logs["prob_loss"],
                        logs["corr_pp_loss"], self.weights_hist[-1]))

            # sample train data for testing
            if self.multiscale:
                valid_size = train_label.size / 3
                rand_ix = np.random.choice(train_label.size,
                        valid_size, replace=False)
                ll_test = train_label[rand_ix]
                dd_test = train_data[rand_ix]
                logger.info("Updating theta ...")
                logger.info("\tBefore update: theta={}, scale_weights={}".format(
                    self.theta, self.scale_weights))

                neg_log_ll = np.zeros_like(self.theta)
                g_losses = np.zeros_like(self.theta)
                for s, scale in enumerate(self.scales):
                    var_mx = None
                    probs = np.zeros(ll_test.size)
                    for i in xrange(ll_test.size):
                        batch_data = util.reshape(dd_test[i], self.win_size, scale)
                        preds_per_ts = self.model.predict_on_batch([batch_data])
                        probs[i] = preds_per_ts[-1].mean()
                        if ll_test[i]==1:
                            flat_gram = np.copy(preds_per_ts[0])
                            if var_mx is None:
                                var_mx = flat_gram 
                            else:
                                var_mx = np.append(var_mx, flat_gram, axis=0)
#                            logger.info("var_mx.shape={}".format(var_mx.shape))

                    if var_mx is not None:
                        g_losses[s] = np.var(var_mx, axis=0).mean()

                    neg_mask = ll_test==0
                    probs[neg_mask] = 1-probs[neg_mask]
                    # although sigmoid should not be 0, numerically it can
                    # to prevent log from outputting -Inf, add a tiny amount
                    if np.any(probs==0):
                        logger.info("Found 0 probability in probs!")
                        probs[probs==0] = 1e-8
                    neg_log_ll[s] = -1.0 * np.mean(np.log(probs))

                logger.info("neg_log_ll={}".format(neg_log_ll))
                scale_prob = np.expand_dims(self.scale_weights, axis=-1)
#                logger.info("scale_prob.shape={}".format(scale_prob.shape))
                scale_jacob = -1.0 * scale_prob.dot(scale_prob.T)
#                logger.info("scale_jacob.shape={}".format(scale_jacob.shape))
                np.fill_diagonal(scale_jacob,
                        self.scale_weights * (1.0 - self.scale_weights))
                grad_theta = 1.0 / T * scale_jacob.dot(neg_log_ll + self.corr_reg * g_losses)
                self.theta -= grad_theta
                logger.info("\tAfter update: theta={}, scale_weights={}".format(
                    self.theta, self.scale_weights))

            logger.info("-"*50)

        train_hist = self.model.fit_generator(
            generator=self.get_rand_batch(train_data, train_label),
            steps_per_epoch=steps_per_epoch,
            callbacks=[
                keras.callbacks.LambdaCallback(
                    on_epoch_end=update_theta),
                ],
            verbose=verbose,
            epochs=self.max_epochs
            )

        logger.info("Saving model ...")
        self.model.save_weights(modelpath)
        if thetapath is not None:
            logger.info("Saving theta to {}".format(thetapath))
            np.savez(thetapath, theta=self.theta)
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
                    "test", 0)
#                    "test", self.downsample)
            for j, scale in enumerate(self.scales):
                batch_data = util.reshape(test_data,
                        self.win_size, scale)
                preds_per_ts = self.model.predict_on_batch([batch_data])
                prob_matrix[i, j] = preds_per_ts[-1].mean()
        probs = prob_matrix.dot(self.scale_weights)
        return probs
