from __future__ import annotations
import os
from unicodedata import decimal
from numpy import dtype
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from src.builders.model_builder import ModelBuilder


class Model2():
    EPOCHS = 222
    BATCH_SIZE = 32
    L1_REGULARIZER = 0.001
    L2_REGULARIZER = 0.0001

    def __init__(self, 
        data: pd.DataFrame, 
        labels: pd.DataFrame,
    ) -> None:
        data.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        self.data = data.to_numpy(dtype="float32")
        self.labels = labels.to_numpy(dtype="float32")
        self.config()
        
    def config(self) -> Model2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def shuffle_data(self) -> Model2:
        dataset = []
        for i, data in enumerate(self.data):
            dataset.append((data, self.labels[i]))
        dataset = shuffle(dataset)
        self.data = np.array([data[0] for data in dataset])
        self.labels = np.array([data[1] for data in dataset])
        return self

    def data_preparation(self) -> Model2: 
        train_count = int(self.data.shape[0] * 0.9)
        self.data_train = self.data[:train_count]
        self.data_test = self.data[train_count:]

        self.labels_train = self.labels[:train_count]
        self.labels_test = self.labels[train_count:]

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.data_train, self.labels_train))
        self.train_dataset = self.train_dataset.batch(Model2.BATCH_SIZE)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.data_test, self.labels_test))
        self.val_dataset = self.val_dataset.batch(Model2.BATCH_SIZE)
        return self

    def build_model(self) -> Model2: 
        input_len = self.data.shape[1]
        self.builder = ModelBuilder() \
            .input(shape=(input_len)) \
            .dense(128, activation='relu') \
            .dropout(0.5) \
            .dense(128, activation='relu') \
            .dropout(0.2) \
            .dense(64, activation='relu') \
            .dropout(0.1) \
            .dense(3) \
            .dense(2) \
            .set_optimizer(tf.keras.optimizers.Adam(learning_rate=1e-3)) \
            .set_loss_fn(tf.keras.losses.CategoricalCrossentropy(from_logits=True)) \
            .set_metric(tf.keras.metrics.AUC(from_logits=True)) \
            .set_validation_metric(tf.keras.metrics.AUC(from_logits=True))
        return self

    def train_model(self) -> Model2: 
        self.builder \
        .build_model() \
        .train(
            self.data_train, 
            self.labels_train,
            self.data_test,
            self.labels_test,
            Model2.BATCH_SIZE,
            Model2.EPOCHS
        )
        return self

    def evaluate_model(self) -> Model2:
        self.builder.get_model().evaluate(
            self.data_test,
            self.labels_test,
            batch_size=Model2.BATCH_SIZE,
            verbose=2,
        )
        # self.builder.load_model(self.builder.best_epoch)
        # val_acc = self.builder.evaluate(
        #     self.val_dataset,
        # )
        # print("Evaluation acc: %.4f" % (float(val_acc),))
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x.reset_index(drop=True, inplace=True)
        x = x.to_numpy()
        prediction = self.builder \
        .softmax() \
        .build_model() \
        .get_model() \
        .predict(x, batch_size=32, verbose=2)
        return pd.DataFrame(
            data={
                'prediction': prediction[:, 1],
            }
        )
        # x.reset_index(drop=True, inplace=True)
        # x = x.to_numpy()
        # self.builder.softmax().build_model()
        # prediction = self.builder \
        #     .softmax() \
        #     .build_model() \
        #     .predict(x, batch_size=Model2.BATCH_SIZE)
        # print(prediction[:10])
        # prediction_np = prediction.to_numpy()[:, 0]
        # prediction_np = np.vectorize(lambda x: 1 - x)(prediction_np)
        # return pd.DataFrame(
        #     data={
        #         'prediction': prediction_np,
        #     }
        # )

    def get_model(self) -> Model2:
        return self.model
