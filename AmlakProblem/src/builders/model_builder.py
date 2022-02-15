from __future__ import annotations
from operator import index
import time
import math
import tensorflow as tf
import numpy as np
import pandas as pd
from src.models.tensor_model import TensorModel



class ModelBuilder:
    def __init__(self) -> None:
        self.input_layer = None
        self.output_layer = None
        self.model = None

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
    
    def set_optimizer(self, optimizer) -> ModelBuilder:
        self.optimizer = optimizer
        return self
    
    def set_loss_fn(self, loss_fn) -> ModelBuilder:
        self.loss_fn = loss_fn
        return self
    
    def set_metric(self, metric) -> ModelBuilder:
        self.metric = metric
        return self
    
    def save_model(self) -> ModelBuilder:
        return self
    
    def train_with_validation(self, train_dataset, val_dataset, epochs) -> ModelBuilder:
        max_auc, index = -1, -1
        for epoch in range(epochs):
            start_time = time.time()
            tup = self.train_epoch(train_dataset)

            val_acc = self.evaluate(val_dataset)
            if val_acc > max_auc:
                max_auc = val_acc
                index = epoch
                self.save_model()
            print("epoch[%02d]: time=%.02fs, traininig_loss=%0.4f, training_auc=%.04f, evaluation_auc=%.04f" % 
            (epoch, time.time() - start_time, tup[1], tup[0], val_acc))
        print(f'the best epoch={index}, best metric={max_auc}')
        return self
    
    @tf.function
    def train_epoch(self, train_dataset) -> tuple:
        model = self.get_model()
        # Iterate over the batches of the dataset.
        loss = 0.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = self.loss_fn(y_batch_train, logits)
                if loss < loss_value:
                    loss = loss_value
                # print(x_batch_train)
                # print(logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            # print(grads)
            self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            self.metric.update_state(y_batch_train, logits)

            # # Log every 200 batches.
            # if step % 200 == 0:
            #     print(
            #         "Training loss (for one batch) at step %d: %.4f"
            #         % (step, float(loss_value))
            #     )
            #     print("Seen so far: %d samples" % ((step + 1) * ModelBuilder.BATCH_SIZE))

        # Display metrics at the end of each epoch.
        train_acc = self.metric.result()
        self.metric.reset_states()
        return train_acc, loss

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
        res = res.reset_index(drop=True)
        # print(res)
        return res

    def get_model(self, fresh=False) -> tf.keras.Model:
        if fresh or self.model is None:
            self.build_model()
        return self.model
    
    def build_model(self) -> ModelBuilder:
        self.model = tf.keras.Model(self.input_layer, self.output_layer)
        return self
