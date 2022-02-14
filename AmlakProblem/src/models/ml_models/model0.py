from __future__ import annotations
from unicodedata import decimal
from numpy import dtype
import numpy as np
import pandas as pd
import tensorflow as tf
import os


class Model0():
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        self.data = data
        self.labels = labels
        self.config()
        
    def config(self) -> Model0:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def data_preparation(self) -> Model0: 
        train_count = int(0.8 * self.data.shape[0])
        self.data = self.data.to_numpy(dtype="float32")
        self.data_train = self.data[:train_count]
        self.data_test = self.data[train_count:]
        self.labels = self.labels.to_numpy(dtype="float32")
        self.labels_train = self.labels[:train_count]
        self.labels_test = self.labels[train_count:]

        print(self.data_train[:10])
        print(self.labels_train[:10])
        return self

    def build_model(self) -> Model0: 
        input_len = self.data.shape[1]
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(input_len)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC()]
        )
        return self

    def train_model(self) -> Model0: 
        self.model.fit(
            x=self.data_train,
            y=self.labels_train,
            batch_size=32,
            epochs=500,
            verbose=2
        )
        return self

    def evaluate_model(self) -> Model0:
        self.model.evaluate(
            self.data_test,
            self.labels_test,
            batch_size=32,
            verbose=2,
        )
        return self

    def predict(self, x: np.array) -> np.array:
        prediction = self.model.predict(x=self.data_train, batch_size=32, verbose=2)
        test = pd.DataFrame()
        test['true-label'] = pd.DataFrame(self.labels_train)
        test['classified-label'] = pd.DataFrame(prediction)
        # print(prediction)
        # prediction = prediction.round(decimals=0)
        print(test[:333])
        return prediction

    