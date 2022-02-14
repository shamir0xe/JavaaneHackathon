from __future__ import annotations
import os
from unicodedata import decimal
from numpy import dtype
import numpy as np
import pandas as pd
import tensorflow as tf
from src.builders.model_builder import ModelBuilder


class Model2():
    EPOCHS = 3
    BATCH_SIZE = 32
    L1_REGULARIZER = 0.001
    L2_REGULARIZER = 0.0001

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        self.data = data
        self.labels = labels
        self.config()
        
    def config(self) -> Model2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def data_preparation(self) -> Model2: 
        train_count = int(self.data.shape[0] * 0.9)
        self.data = self.data.to_numpy(dtype="float32")
        self.data_train = self.data[:train_count]
        self.data_test = self.data[train_count:]
        self.labels = self.labels.to_numpy(dtype="float32")
        self.labels = np.insert(self.labels, axis=1, obj=1, values=[(1 ^ round(float(x))) for x in self.labels[:, 0]])
        self.labels_train = self.labels[:train_count]
        self.labels_test = self.labels[train_count:]

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.data_train, self.labels_train))
        self.train_dataset = self.train_dataset.batch(Model2.BATCH_SIZE)
        # self.train_dataset = self.train_dataset.shuffle(buffer_size=1024).batch(Model2.BATCH_SIZE)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.data_test, self.labels_test))
        self.val_dataset = self.val_dataset.batch(Model2.BATCH_SIZE)
        return self

    def build_model(self) -> Model2: 
        input_len = self.data.shape[1]
        self.builder = ModelBuilder() \
            .input(shape=(input_len)) \
            .dense(256, activation='relu') \
            .dropout(0.5) \
            .dense(128, activation='relu') \
            .dropout(0.2) \
            .dense(2) \
            .set_optimizer(tf.keras.optimizers.Adam()) \
            .set_loss_fn(tf.keras.losses.CategoricalCrossentropy(from_logits=True)) \
            .set_metric(tf.keras.metrics.AUC(from_logits=True))
        return self

    def train_model(self) -> Model2: 
        self.builder.train_with_validation(self.train_dataset, self.val_dataset, epochs=Model2.EPOCHS)
        return self

    def evaluate_model(self) -> Model2:
        val_acc = self.builder.evaluate(
            self.val_dataset,
        )
        print("Evaluation acc: %.4f" % (float(val_acc),))
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.to_numpy(na_value=0.)
        model = self.builder.softmax().get_model(True)
        prediction = self.builder.predict(x, batch_size=Model2.BATCH_SIZE)
        return pd.DataFrame(
            data={
                'prediction': prediction.to_numpy()[:, 0],
            }
        )

    def get_model(self) -> Model2:
        return self.model
