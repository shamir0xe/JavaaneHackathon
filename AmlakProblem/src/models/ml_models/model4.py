from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from src.models.data_types import DataTypes
from src.helpers.data_helper import DataHelper
from src.builders.model_builder import ModelBuilder
from src.helpers.tensorflow_helper import TensorflowHelper


class Model4():
    EPOCHS = 55
    BATCH_SIZE = 32
    L1_REGULARIZER = 0.001
    L2_REGULARIZER = 0.0001

    def __init__(self, 
        data: pd.DataFrame, 
        labels: pd.DataFrame,
    ) -> None:
        data.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        self.data = data
        self.labels = labels.to_numpy(dtype="float32")
        self.config()
        
    def config(self) -> Model4:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def shuffle_data(self) -> Model4:
        dataset = []
        count_ones = 0
        for j, (i, data) in enumerate(self.data.iterrows()):
            if j != i:
                raise Exception('asshole')
            if self.labels[i, 0] == 1:
                count_ones += 1
            dataset.append((data, self.labels[i]))
        dataset = shuffle(dataset)
        dataset_prim = []
        count = int(len(dataset) * 0.66) - count_ones
        for data in dataset:
            if data[1][0] == 1 and count > 0:
                count -= 1
                continue
            dataset_prim.append(data)
        dataset = dataset_prim
        self.data = pd.DataFrame([data[0] for data in dataset], columns=self.data.columns)
        self.data.reset_index(drop=True, inplace=True)
        self.labels = np.array([data[1] for data in dataset])
        return self

    def pack_flows(self, data: pd.DataFrame) -> dict:
        flows = {}
        flows[DataTypes.IMAGE.value] = DataHelper.pack_flow(data, DataTypes.IMAGE)
        flows[DataTypes.TEXT.value] = DataHelper.pack_flow(data, DataTypes.TEXT)
        flows[DataTypes.CATEGORICAL.value] = DataHelper.pack_flow(data, DataTypes.CATEGORICAL)
        flows[DataTypes.NUMERICAL.value] = DataHelper.pack_flow(data, DataTypes.NUMERICAL)
        return flows

    def data_preparation(self) -> Model4: 
        self.train_count = int(self.data.shape[0] * 0.85)
        self.flows_train = self.pack_flows(self.data[:self.train_count])
        self.flows_validation = self.pack_flows(self.data[self.train_count:])

        # self.data_train = self.data[:train_count]
        # self.labels_train = self.labels[:train_count]

        # self.data_test = self.data[train_count:]
        # self.labels_test = self.labels[train_count:]
        return self

    def build_image(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.IMAGE.value]
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.IMAGE.value) \
            .dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)) \
            .dropout(0.2) \
            .dense(64, activation='relu')
        self.inputs.append(builder.get_input())
        return builder

    def build_text(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.TEXT.value]
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.TEXT.value) \
            .dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)) \
            .dropout(0.2) \
            .dense(64, activation='relu')
        self.inputs.append(builder.get_input())
        return builder

    def build_categorical(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.CATEGORICAL.value]
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.CATEGORICAL.value) \
            .dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)) \
            .dropout(0.2) \
            .dense(64, activation='relu') \
            .dense(32, activation='relu') \
            .dense(2, activation='relu') \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_numerical(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.NUMERICAL.value]
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.NUMERICAL.value) \
            .dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)) \
            .dropout(0.2) \
            .dense(64, activation='relu') \
            .dense(32, activation='relu') \
            .dense(2, activation='relu') \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_graph(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.GRAPH.value]
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.GRAPH.value) \
            .dense(128, activation='relu') \
            .dropout(0.2) \
            .dense(64, activation='relu') \
            .dense(32, activation='relu') \
            .dense(2, activation='relu') \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder
    
    def build_model(self) -> Model4: 
        self.builders = {}
        self.inputs = []
        image_net = self.build_image()
        text_net = self.build_text()
        self.builders['IMAGE_TEXT'] = ModelBuilder() \
            .concat([image_net.get_output(), text_net.get_output()]) \
            .dense(64, activation='relu') \
            .dropout(0.2) \
            .dense(32, activation='relu') \
            .dense(2) \
            .softmax()
        self.builders[DataTypes.CATEGORICAL.value] = self.build_categorical()
        self.builders[DataTypes.NUMERICAL.value] = self.build_numerical()
        # self.builders[DataTypes.GRAPH.value] = self.build_graph()

        self.builder = ModelBuilder() \
            .add([builder.get_output() for _, builder in self.builders.items()]) \
            .dense(2) \
            .set_optimizer(tf.keras.optimizers.Adam(learning_rate=1e-4)) \
            .set_loss_fn(tf.keras.losses.CategoricalCrossentropy()) \
            .set_metric(tf.keras.metrics.AUC(from_logits=True))
        return self

    def train_model(self) -> Model4: 
        self.builder \
        .softmax() \
        .build_model(inputs=self.inputs) \
        .plot()
        self.builder.train(
            self.flows_train, 
            self.labels[:self.train_count],
            self.flows_validation,
            self.labels[self.train_count:],
            Model4.BATCH_SIZE,
            Model4.EPOCHS
        )
        return self

    def evaluate_model(self) -> Model4:
        self.builder.get_model().evaluate(
            self.flows_validation,
            self.labels[self.train_count:],
            batch_size=Model4.BATCH_SIZE,
            verbose=2,
        )
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x.reset_index(drop=True, inplace=True)
        x_flow = self.pack_flows(x)
        prediction = self.builder \
        .build_model(inputs=self.inputs) \
        .get_model() \
        .predict(x_flow, batch_size=32, verbose=2)
        return pd.DataFrame(
            data={
                'prediction': prediction[:, 1],
            }
        )

    def get_model(self) -> Model4:
        return self.model
