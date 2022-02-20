import tensorflow as tf
import numpy as np
import tflearn


class TensorflowHelper:
    @staticmethod
    def auc_loss_fn(y_true, y_pred):
        return tflearn.objectives.roc_auc_score(y_pred, y_true)

