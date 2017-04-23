import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Dense, Flatten, Lambda, Reshape, MaxPooling1D
from keras.models import Model
from keras.losses import binary_crossentropy
from sklearn import metrics
import pandas as pd
import numpy as np
import threading
import os
import logging
logger = logging.getLogger(__name__)


NUM_FETCHER = 4
MAX_Q_SIZE = 16


class TsNet:

    def __init__(self, args, train_mean, train_std,
            train_data, train_label, valid_data, valid_label):

        self.win_size = args.win_size
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.corr_coef_pp = args.corr_coef_pp
        self.log_n_iter = args.log_n_iter

        self.train_mean = train_mean
        self.train_std = train_std
        self.train_data = train_data
        self.train_label = train_label
        self.valid_data = valid_data
        self.valid_label= valid_label
        self.num_channel = train_data.shape[-1]

        self.should_stop = False

    def build_model(self):

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
        
        # 1D conv
        conv_params = [(32, 7, 4), (64, 3, 2), (64, 3, 2)]
        n_convs = len(conv_params)
        loss_weights = [1.0 * self.corr_coef_pp / n_convs] * n_convs
        corr_layer = Lambda(corr_pp, name="corr_pp")
        for i, conv_param in enumerate(conv_params):
            num_filter, filter_size, pool_size = conv_param
            conv_layer = Conv1D(num_filter, filter_size, activation="relu", padding="same", name="conv" + str(i+1))
            maxpool_layer = MaxPooling1D(pool_size=pool_size, strides=pool_size)
            x_all_data = conv_layer(x_all_data)
            x_pos_data1 = conv_layer(x_pos_data1)
            x_pos_data2 = conv_layer(x_pos_data2)
            corr_pp = corr_layer([x_pos_data1, x_pos_data2])
            x_all_data = maxpool_layer(x_all_data)
            x_pos_data1 = maxpool_layer(x_pos_data1)
            x_pos_data2 = maxpool_layer(x_pos_data2)
            outputs.append(corr_pp)
            losses.append(corr_loss_func)

        # fc layers
        x_all_data = Flatten(name="flatten")(x_all_data)
        x_all_data = Dense(1024, activation="relu", name="fc1")(x_all_data)
        x_all_data = Dense(1024, activation="relu", name="fc2")(x_all_data)
        prob = Dense(1, activation="sigmoid", name="prob")(x_all_data)
        
        outputs.append(prob)
        losses.append(binary_crossentropy)
        loss_weights.append(1.0)
        
        self.num_outputs = len(outputs)
        logger.info(outputs)
        logger.info(loss_weights)
        self.model = Model(inputs=[input_data, pos_data1, pos_data2],
                outputs=outputs)
        self.model.compile(optimizer="adam", loss=losses, loss_weights=loss_weights)

    def build_func(self):
        self.get_prob = K.function([self.model.layers[0].input],
                [self.model.layers[14].output])

    def fetch_batch_data(self, data, labels):

        data_size = data.shape[0]
        seq_length = data.shape[1]
        pos_label_idx = np.where(labels==1)[0]
        
        num_pos_samples = np.sum(labels==1)
        num_neg_samples = np.sum(labels==0)
        if 1.0 * num_neg_samples / num_pos_samples > 2:
            pos_sampling_weight = num_neg_samples / (num_pos_samples * 2.0)
        else:
            pos_sampling_weight = 1.0
        sampling_weights = np.ones(data_size)
        sampling_weights[labels==1] *= pos_sampling_weight
        sampling_weights /= np.sum(sampling_weights)
        logger.info("num_pos_samples={}, num_neg_samples={}".format(
            num_pos_samples, num_neg_samples))
        logger.info("class_sampling_weights={}".format(
            [1.0, pos_sampling_weight]))

        logger.info("data_fetcher off to work")
        while not self.should_stop:
            sample_idx = np.random.choice(data_size, self.batch_size,
                    p=sampling_weights)
            batch_label = labels[sample_idx]
            sample_idx = np.expand_dims(sample_idx, -1)
            pos_idx1 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)
            pos_idx2 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)
            rand_start = np.random.randint(0, seq_length - self.win_size,
                    self.batch_size)
            rand_end = rand_start + self.win_size
            seq_slice = np.array([np.arange(rand_start[i], rand_end[i])
                for i in xrange(self.batch_size)])
            batch_data = data[sample_idx, seq_slice]
            batch_pos_data1 = data[pos_idx1, seq_slice]
            batch_pos_data2 = data[pos_idx2, seq_slice]
            yield ([batch_data, batch_pos_data1, batch_pos_data2],
                    [batch_label]*self.num_outputs)
        logger.info("data_fetcher abort") 

    def train(self, logdir, modelpath):

        def get_steps(data):
            samples_per_epoch = np.prod(data.shape[:-1])
            samples_per_batch = self.win_size * self.batch_size
            steps_per_epoch =  samples_per_epoch / samples_per_batch
            return steps_per_epoch

        steps_per_epoch = get_steps(self.train_data)
        validation_steps = get_steps(self.valid_data)
        logger.info("max_epochs={}, steps_per_epoch={}, validation_steps={}".format(
            self.max_epochs, steps_per_epoch, validation_steps))

        train_hist = self.model.fit_generator(
            generator=self.fetch_batch_data(
                self.train_data, self.train_label),
            steps_per_epoch=steps_per_epoch,
            validation_data=self.fetch_batch_data(
                self.valid_data, self.valid_label),
            validation_steps=validation_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0.001, patience=3, verbose=0, mode='min'),
                keras.callbacks.ModelCheckpoint(modelpath,
                    monitor='val_loss',
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    period=1)
                ],
            verbose=0,
            epochs=self.max_epochs
            )
        self.should_stop = True

        return train_hist

    def test_on_data(self, test_data):
        """Test model performance on the test set""" 
        test_size, ts_len, num_channel = test_data.shape
        preds = np.zeros(test_size)
        num_pred_per_ts = 600
        stride = int((ts_len - self.win_size) / (num_pred_per_ts - 1))
        
        for i in xrange(test_size):
            preds_per_ts = np.zeros(num_pred_per_ts)
            offset = 0
            remaining_num = num_pred_per_ts
            while remaining_num > 0:
                # segment
                test_batch_size = np.min([remaining_num, self.batch_size])
                start_idx = np.arange(offset, offset+test_batch_size*stride,
                        stride)
                end_idx = start_idx + self.win_size
                slice_idx = np.array([np.arange(start_idx[k], end_idx[k])
                    for k in xrange(test_batch_size)])
                batch_data = test_data[i, slice_idx]
                # get test scores
                probs = self.get_prob([batch_data])[0].flatten()
                preds_per_ts[offset:(offset+test_batch_size)] =\
                        probs[:test_batch_size]
                offset += test_batch_size
                remaining_num -= test_batch_size
            preds[i] = preds_per_ts.mean()
        return preds

#    def train(self, logdir):
#        # start data fetcher
#        tt = []
#        for _ in xrange(NUM_FETCHER):
#            th = threading.Thread(target=self.fetch_batch_data)
#            th.setDaemon(True)
#            th.start()
#            tt.append(th)
#
#        losses = np.zeros(self.max_iter+1)
#        prob_losses = np.zeros(self.max_iter+1)
#        corr_losses = np.zeros(self.max_iter+1)
#        for n_iter in xrange(self.max_iter+1):
#            data_batch, label_batch, pos_batch1, pos_batch_2 = self.fifo_q.get()
#            res = self.model.train_on_batch([data_batch, pos_batch1, pos_batch_2],
#                    [label_batch]*self.num_outputs)
#            loss = res[0]
#            prob_loss = res[-1]
#            corr_loss = np.sum(res[1:-1])
#            losses[n_iter] = loss
#            prob_losses[n_iter] = prob_loss
#            corr_losses[n_iter] = corr_loss
#            
#            if n_iter % self.log_n_iter == 0:
#                logger.info("iter={0}, loss={1:.6f}, "\
#                        "prob_loss={2:.6f}, corr_loss={3:.6f}".format(
#                            n_iter, loss, prob_loss, corr_loss))
#                logger.info("--------------")
#        
#        # record losses
#        train_stat = pd.DataFrame({"loss": losses,
#            "prob_loss": prob_losses,
#            "corr_loss": corr_losses}, index=np.arange(self.max_iter+1))
#        train_stat.to_csv(os.path.join(logdir, "train_stat.csv"))
#
#        logger.info("Training done.")

    def save(self, logdir):
        self.model.save(os.path.join(logdir, "model", "ts_net.h5"))
        logger.info("Model saved to {}.".format(logdir)) 
