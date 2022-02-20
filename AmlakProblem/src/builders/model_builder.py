from __future__ import annotations
import enum
from operator import index
import time
import math
from typing import Any
import tensorflow as tf
import numpy as np
import pandas as pd
from src.helpers.data_helper import DataHelper
from src.helpers.memory_helper import MemoryHelper
from src.models.tensor_model import TensorModel


class ModelBuilder:
    def __init__(self) -> None:
        self.input_layer = None
        self.output_layer = None
        self.model = None
        self.metrics_train = []
        self.metrics_val = []

    def input(self, *args, **kwargs) -> ModelBuilder:
        self.input_layer = tf.keras.layers.Input(*args, **kwargs)
        return self
    
    def dense(self, *args, **kwargs) -> ModelBuilder:
        if self.output_layer is None:
            self.output_layer = tf.keras.layers.Dense(*args, **kwargs)(self.input_layer)
        else:
            self.output_layer = tf.keras.layers.Dense(*args, **kwargs)(self.output_layer)
        return self
    
    def dropout(self, *args, **kwargs) -> ModelBuilder:
        if self.output_layer is None:
            self.output_layer = tf.keras.layers.Dropout(*args, **kwargs)(self.input_layer)
        else:
            self.output_layer = tf.keras.layers.Dropout(*args, **kwargs)(self.output_layer)
        return self
    
    def softmax(self, *args, **kwargs) -> ModelBuilder:
        if self.output_layer is None:
            self.output_layer = tf.keras.layers.Softmax(*args, **kwargs)(self.input_layer)
        else:
            self.output_layer = tf.keras.layers.Softmax(*args, **kwargs)(self.output_layer)
        return self
    
    def concat(self, *args, **kwargs) -> ModelBuilder:
        self.output_layer = tf.keras.layers.Concatenate()(*args, **kwargs)
        return self
    
    def add(self, *args, **kwargs) -> ModelBuilder:
        self.output_layer = tf.keras.layers.Add()(*args, **kwargs)
        return self
    
    def avg(self, *args, **kwargs) -> ModelBuilder:
        self.output_layer = tf.keras.layers.Average()(*args, **kwargs)
        return self

    def get_output(self) -> Any:
        return self.output_layer

    def get_input(self) -> Any:
        return self.input_layer
    
    def set_optimizer(self, optimizer) -> ModelBuilder:
        self.optimizer = optimizer
        return self
    
    def set_loss_fn(self, loss_fn) -> ModelBuilder:
        self.loss_fn = loss_fn
        return self
    
    def set_metric(self, metric) -> ModelBuilder:
        self.train_metric = metric
        return self
    
    def set_validation_metric(self, metric) -> ModelBuilder:
        self.val_metric = metric
        return self
    
    def save_model(self, epoch: int) -> ModelBuilder:
        self.get_model().save_weights(DataHelper.cache_path(f'tf_model_{epoch}'))
        # print('model saved at epoch %d' % epoch)
        return self
    
    def load_model(self, epoch: int=None) -> ModelBuilder:
        if epoch is None:
            epoch = self.best_epoch
        self.get_model().load_weights(DataHelper.cache_path(f'tf_model_{epoch}'))
        print('model retrieved at epoch %d' % epoch)
        return self
    
    def train_with_validation(self, train_dataset, val_dataset, epochs) -> ModelBuilder:
        val_metric, index = -1, -1
        for epoch in range(epochs):
            start_time = time.time()
            loss = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                if step == 0:
                    loss = loss_value
                else:
                    loss = loss + 0.1 * (loss_value - loss)
                # Log every 200 batches.
                # if step % 200 == 0:
                #     print(
                #         "Training loss (for one batch) at step %d: %.4f"
                #         % (step, float(loss_value))
                #     )
                #     print("Seen so far: %d samples" % ((step + 1) * batch_size))

            train_metric = self.train_metric.result()
            self.train_metric.reset_states()
            for x_batch_val, y_batch_val in val_dataset:
                self.test_step(x_batch_val, y_batch_val)
            val_metric = self.val_metric.result()
            self.val_metric.reset_state()
            if index == -1 or val_metric > max_val:
                max_val = val_metric
                index = epoch
                self.save_model(index)
            print("epoch[%3d/%3d]: time %.02fs, train_loss %0.4f, train_auc %.04f, eval_auc %.04f" % 
                (epoch, epochs, time.time() - start_time, loss, train_metric, val_metric))

        self.best_epoch = index
        print(f'the best epoch {index}, best metric {val_metric}')
        return self
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.get_model()(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.get_model().trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.get_model().trainable_weights))
        self.train_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.get_model()(x, training=False)
        self.val_metric.update_state(y, val_logits)

    @tf.function
    def evaluate(self, val_dataset) -> float:
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = self.get_model()(x_batch_val, training=False)
            # Update val metrics

            self.metric.update_state(y_batch_val, val_logits)
        val_acc = self.metric.result()
        self.metric.reset_states()
        return val_acc
    
    def predict(self, val_dataset, batch_size) -> pd.DataFrame:
        m = math.ceil(val_dataset.shape[0] / batch_size - 1e-9)
        res = pd.DataFrame()
        # print(f'm ine: {m}')
        for i in range(m):
            outer_range = min((i + 1) * batch_size, val_dataset.shape[0])
            output = self.get_model()(val_dataset[i * batch_size:outer_range,:], training=False)
            res = pd.concat([res, pd.DataFrame(output)], axis=0)
        # res = res.drop(res.columns[[0]], axis=1)
        # print('panda ine')
        res.reset_index(drop=True, inplace=True)
        # print(res)
        return res

    def get_model(self, fresh=False) -> tf.keras.Model:
        if fresh or self.model is None:
            self.build_model()
        return self.model
    
    def plot(self) -> ModelBuilder:
        tf.keras.utils.plot_model(self.get_model(), show_shapes=True)
        return self

    def build_model(self, inputs: list=[], outputs: list=[]) -> ModelBuilder:
        if len(inputs) == 0:
            inputs = [self.input_layer]
        if len(outputs) == 0:
            outputs = [self.output_layer]
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return self
    
    def train(self, data_train, labels_train, data_test, labels_test, batch_size, epochs) -> ModelBuilder:
        self.compile()
        self.fit(data_train, labels_train, data_test, labels_test, batch_size, epochs)
        return self
    
    def compile(self) -> ModelBuilder:
        self._initialize_metrics()
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=[self.train_metric]
        )
    
    def fit(self, data_train, labels_train, data_test, labels_test, batch_size, epochs) -> ModelBuilder:
        self.model.fit(
            x = data_train,
            y = labels_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(data_test, labels_test),
            callbacks=[BestValidation(self)]
        )
        return self
    
    def plotlib(self, model: str='metric_epoch') -> ModelBuilder:
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os
        plt.plot(np.arange(len(self.metrics_train)), np.array(self.metrics_train), color='red', label='auc')
        plt.plot(np.arange(len(self.metrics_val)), np.array(self.metrics_val), color='blue', label='val-auc')
        plt.legend(loc='upper left')
        path = DataHelper.data_path('images/%s.png' % datetime.now().strftime("%d-%b-%Y (%H_%M_%S)"))
        plt.savefig(path)
        plt.show()
        return self
    
    def _initialize_metrics(self) -> None:
        self.metrics_train = []
        self.metrics_val = []

class BestValidation(tf.keras.callbacks.Callback):
    def __init__(self, model_builder: ModelBuilder):
        super(BestValidation, self).__init__()
        self.builder = model_builder
        self.best_metric = None
        self.train_metric = []
        self.val_metric = []

    def on_epoch_end(self, epoch, logs=None):
        # print('EPOCH: ', epoch)
        # print('HOHOHO: ', logs)
        auc_key = None
        val_auc_key = None
        keys = logs.keys()
        for key in keys:
            if key.startswith('auc'):
                auc_key = key
            if key.startswith('val_auc'):
                val_auc_key = key
        if self.best_metric is None or logs[val_auc_key] > self.best_metric:
            self.best_metric = logs[auc_key]
            self.builder.best_epoch = epoch
            self.builder.save_model(epoch=epoch)
        self.builder.metrics_train.append(logs[auc_key])
        self.builder.metrics_val.append(logs[val_auc_key])
