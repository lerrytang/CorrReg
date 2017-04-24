import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Dense, Flatten, Lambda, MaxPooling1D
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import optimizers
import util
import pandas as pd
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


class TsNet:

    def __init__(self, args, train_mean, train_std, num_channel):

        self.win_size = args.win_size
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.corr_coef_pp = args.corr_coef_pp
        self.test_data_dir = os.path.join(args.data_dir, args.target_obj)
        self.train_mean = train_mean
        self.train_std = train_std
        self.num_channel = num_channel
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
        conv_params = [(32, 7, 5), (64, 5, 2), (64, 3, 2), (64, 3, 2)]
        n_convs = len(conv_params)
        loss_weights = [1.0 * self.corr_coef_pp / n_convs] * n_convs
        corr_layer = Lambda(corr_pp, name="corr_pp")
        for i, conv_param in enumerate(conv_params):
            num_filter, filter_size, pool_size = conv_param
            conv_layer = Conv1D(num_filter, filter_size,
                    activation="relu", padding="same", name="conv" + str(i+1))
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
        optimizer = optimizers.Adam(lr=0.001, decay=1e-6)
        self.model.compile(optimizer=optimizer, loss=losses,
                loss_weights=loss_weights)

    def train(self, train_data, train_label, valid_data, valid_label,
            logdir, modelpath, verbose=0):

        samples_per_epoch = np.prod(train_data.shape[:-1])
        samples_per_batch = self.win_size * self.batch_size
        steps_per_epoch =  samples_per_epoch / samples_per_batch
        steps_per_epoch = 2000
        logger.info("max_epochs={}, steps_per_epoch={}".format(
            self.max_epochs, steps_per_epoch))

        train_hist = self.model.fit(x=[train_data]*3,
                y=[train_label]*self.num_outputs,
                batch_size=self.batch_size,
                epochs=self.max_epochs,
                verbose=verbose,
                validation_data=([valid_data]*3,
                    [valid_label]*self.num_outputs),
                callbacks=[
                    keras.callbacks.ModelCheckpoint(modelpath,
                        monitor='val_prob_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min',
                        period=1)
                    ])
        return train_hist

    def test_on_data(self, test_data_files):
        """Test model performance on the test set""" 
        preds = np.zeros(test_data_files.size)
        for i, f in enumerate(test_data_files):
            test_data, _, _, _, _ = \
                    util.load_data(os.path.join(self.test_data_dir, f),
                            self.win_size, "test")
            preds_per_ts = self.model.predict_on_batch([test_data] * 3)
            logger.info(preds_per_ts)
            preds[i] = preds_per_ts[-1].mean()
        return preds
