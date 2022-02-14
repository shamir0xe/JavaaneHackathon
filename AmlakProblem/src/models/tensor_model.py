import tensorflow as tf
import numpy as np


class TensorModel(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def train_step(self, data: np.array):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        print(f'yolo EPOCH?, data = {x}')
        print(f'yolo EPOCH?, data = {(x, y)}')
        print(f'yolo EPOCH?, data = {(x, y)}')
        print(f'yolo EPOCH?, data = {(x, y)}')
        print(f'yolo EPOCH?, data = {(x, y)}')
        print(f'yolo EPOCH?, data = {(x, y)}')
        print(f'yolo EPOCH?, data = {y}')
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
