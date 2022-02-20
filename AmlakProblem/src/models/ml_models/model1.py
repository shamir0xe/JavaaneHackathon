from __future__ import annotations
from unicodedata import decimal
from numpy import dtype
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import os


class Model1():
    EPOCHS = 333
    BATCH_SIZE = 32
    L1_REGULARIZER = 0.001
    L2_REGULARIZER = 0.0001

    def __init__(self, 
        data: pd.DataFrame, 
        labels: pd.DataFrame,
        class_cnt: int=2
    ) -> None:
        data.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        self.data = data.to_numpy(dtype="float32")
        self.labels = labels.to_numpy(dtype="float32")
        self.config()
        
    def config(self) -> Model1:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def shuffle_data(self) -> Model1:
        dataset = []
        for i, data in enumerate(self.data):
            dataset.append((data, self.labels[i]))
        dataset = shuffle(dataset)
        self.data = np.array([data[0] for data in dataset])
        self.labels = np.array([data[1] for data in dataset])
        return self

    def data_preparation(self) -> Model1: 
        train_count = int(self.data.shape[0] * 0.9)
        self.data_train = self.data[:train_count]
        self.data_test = self.data[train_count:]

        self.labels_train = self.labels[:train_count]
        self.labels_test = self.labels[train_count:]
        return self

    def build_model(self) -> Model1: 
        input_len = self.data.shape[1]
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(input_len)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(3),
            tf.keras.layers.Dense(2),
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.AUC(from_logits=True)]
        )
        return self

    def train_model(self) -> Model1: 
        # print(self.data_train[:1])
        # print(self.labels_train[:1])
        self.model.fit(
            x=self.data_train,
            y=self.labels_train,
            batch_size=Model1.BATCH_SIZE,
            epochs=Model1.EPOCHS,
            verbose=2,
            validation_data=(self.data_test, self.labels_test)
        )
        return self

    def evaluate_model(self) -> Model1:
        # print(self.data_test[:1])
        # print(self.labels_test[:1])
        self.model.evaluate(
            self.data_test,
            self.labels_test,
            batch_size=Model1.BATCH_SIZE,
            verbose=2,
        )
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x.reset_index(drop=True, inplace=True)
        x = x.to_numpy()
        model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        prediction = model.predict(x, batch_size=32, verbose=2)
        # prediction = self.model.predict(x=self.data_train, batch_size=32, verbose=2)
        # test = pd.DataFrame()
        # test['true-label'] = pd.DataFrame(self.labels_train[:, 0])
        # test['classified-label'] = pd.DataFrame(prediction[:, 0])
        # print(test[:333])
        # print(prediction)
        # prediction = prediction.round(decimals=0)
        # print(prediction[:10])
        return pd.DataFrame(
            data={
                'prediction': prediction[:, 1],
            }
        )

    def get_model(self) -> Model1:
        return self.model

    