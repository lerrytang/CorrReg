import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Dense, Flatten, Lambda, MaxPooling1D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import regularizers, optimizers
import util
import pandas as pd
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


class TsNet:

    def __init__(self, args, train_mean, train_std, num_channel):

        self.reg_coef = args.reg_coef
        self.lr = args.init_lr
        self.win_size = args.win_size
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.corr_coef_pp = args.corr_coef_pp
        self.test_data_dir = os.path.join(args.data_dir, args.target_obj)
        self.train_mean = train_mean
        self.train_std = train_std
        self.num_channel = num_channel

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
        conv_params = [(32, 7, 5), (64, 5, 2), (64, 3, 2), (64, 3, 2)]
        n_convs = len(conv_params)
        loss_weights = [1.0 * self.corr_coef_pp / n_convs] * n_convs
        corr_layer = Lambda(corr_pp, name="corr_pp")
        for i, conv_param in enumerate(conv_params):
            num_filter, filter_size, pool_size = conv_param
            conv_layer = Conv1D(num_filter, filter_size,
                    activation="relu", padding="same",
                    kernel_regularizer=regularizers.l2(self.reg_coef),
                    name="conv" + str(i+1))
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
        prob = Dense(1, activation="sigmoid",
                kernel_regularizer=regularizers.l2(self.reg_coef),
                name="prob")(x_all_data)
        
        outputs.append(prob)
        losses.append(binary_crossentropy)
        loss_weights.append(1.0)
        
        self.num_outputs = len(outputs)
        logger.info(outputs)
        logger.info(loss_weights)
        self.model = Model(inputs=[input_data, pos_data1, pos_data2],
                outputs=outputs)
        optimizer = optimizers.Adam(lr=self.lr, decay=1e-6)
        self.model.compile(optimizer=optimizer,
                loss=losses, loss_weights=loss_weights)

    def get_rand_batch(self, data, labels):
        data_size, seq_len, _ = data.shape
        pos_label_idx = np.where(labels==1)[0]
        num_pos_samples = np.sum(labels==1)
        num_neg_samples = np.sum(labels==0)
        pos_sampling_weight = 1.0 * num_neg_samples / num_pos_samples
        logger.info("pos_sampling_weight={}".format(
            pos_sampling_weight))
        sampling_weights = np.ones(data_size)
        sampling_weights[labels==1] *= pos_sampling_weight
        sampling_weights /= np.sum(sampling_weights)

        logger.info("data_fetcher off to work")
        while True:
            sample_idx = np.random.choice(data_size, self.batch_size,
                    p=sampling_weights)
            batch_label = labels[sample_idx]
            sample_idx = np.expand_dims(sample_idx, -1)
            pos_idx1 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)
            pos_idx2 = np.expand_dims(
                    np.random.choice(pos_label_idx, self.batch_size), -1)
            rand_start = np.random.randint(0, seq_len - self.win_size,
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

    def get_ordered_batch(self, data, labels):
        data_size, seq_len, _ = data.shape
        while True:
            for i in xrange(data_size):
                batch_data = util.reshape(data[i], self.win_size)
                batch_label = np.ones(batch_data.shape[0]) * labels[i]
                yield ([batch_data]*3, [batch_label]*self.num_outputs) 
    
    def train(self, train_data, train_label, valid_data, valid_label,
            logdir, modelpath, verbose=0):

        samples_per_epoch = np.prod(train_data.shape[:-1])
        samples_per_batch = self.win_size * self.batch_size
        steps_per_epoch =  samples_per_epoch / samples_per_batch
        validation_steps = valid_data.shape[0]
        logger.info("max_epochs={}, steps_per_epoch={}, "\
                "validation_steps={}".format(
                    self.max_epochs, steps_per_epoch, validation_steps))

        train_hist = self.model.fit_generator(
            generator=self.get_rand_batch(train_data, train_label),
            steps_per_epoch=steps_per_epoch,
            validation_data=self.get_ordered_batch(valid_data, valid_label),
            validation_steps=validation_steps,
            callbacks=[
#                keras.callbacks.EarlyStopping(monitor='val_prob_loss',
#                    min_delta=0.0001,
#                    patience=5,
#                    verbose=1,
#                    mode='min'),
                keras.callbacks.ModelCheckpoint(modelpath,
                    monitor='val_prob_loss',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='min',
                    period=1)
                ],
            verbose=verbose,
            epochs=self.max_epochs
            )
        return train_hist

    def test_on_data(self, test_data_files):
        """Test model performance on the test set"""
        preds = np.zeros(test_data_files.size)
        for i, f in enumerate(test_data_files):
            test_data = util.load_data_for_test(
                    os.path.join(self.test_data_dir, f), self.win_size)
            preds_per_ts = self.model.predict_on_batch([test_data] * 3)
            preds[i] = preds_per_ts[-1].mean()
        return preds
