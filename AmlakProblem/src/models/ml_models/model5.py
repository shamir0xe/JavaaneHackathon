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


class Model5():
    EPOCHS = 555
    BATCH_SIZE = 32
    L1_REGULARIZER = 0.001
    L2_REGULARIZER = 0.01

    def __init__(self, 
        data: pd.DataFrame, 
        labels: pd.DataFrame,
        class_cnt: int=2
    ) -> None:
        data.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        self.data = data
        self.labels = labels.to_numpy(dtype="float32")
        self.class_cnt = class_cnt
        self.config()
        
    def config(self) -> Model5:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        print(f'gpus: {tf.config.list_physical_devices("GPU")}')
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices('GPU')[0], 
            True
        )
        return self

    def shuffle_data(self) -> Model5:
        dataset = []
        count_ones = 0
        for j, (i, data) in enumerate(self.data.iterrows()):
            if j != i:
                raise Exception('asshole')
            if self.labels[i, 0] == 1:
                count_ones += 1
            dataset.append((data, self.labels[i]))
        dataset = shuffle(dataset)
        # dataset_prim = []
        # count = int(len(dataset) * 0.66) - count_ones
        # for data in dataset:
        #     if data[1][0] == 1 and count > 0:
        #         count -= 1
        #         continue
        #     dataset_prim.append(data)
        # dataset = dataset_prim
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
        flows[DataTypes.DUPLICATE_PIC.value] = DataHelper.pack_flow(data, DataTypes.DUPLICATE_PIC)
        return flows

    def data_preparation(self) -> Model5: 
        self.train_count = int(self.data.shape[0] * 0.75)
        self.flows_train = self.pack_flows(self.data[:self.train_count])
        self.flows_validation = self.pack_flows(self.data[self.train_count:])

        # self.data_train = self.data[:train_count]
        # self.labels_train = self.labels[:train_count]

        # self.data_test = self.data[train_count:]
        # self.labels_test = self.labels[train_count:]
        return self

    def func(self) -> Any:
        return lambda x: DataHelper.max_pow_2(x) * 2

    def build_image(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.IMAGE.value]
        m = self.func()(flow.shape[1])
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.IMAGE.value) \
            .dense(m, activation='relu') \
            .dropout(0.2) \
            .dense(int(m / 2)) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_text(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.TEXT.value]
        m = self.func()(flow.shape[1])
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.TEXT.value) \
            .dense(m, activation='relu') \
            .dropout(0.2) \
            .dense(int(m / 2)) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_categorical(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.CATEGORICAL.value]
        m = self.func()(flow.shape[1])
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.CATEGORICAL.value) \
            .dense(m, activation='relu') \
            .dropout(0.2) \
            .dense(int(m / 2)) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_numerical(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.NUMERICAL.value]
        m = self.func()(flow.shape[1])
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.NUMERICAL.value) \
            .dense(m, activation='relu') \
            .dropout(0.2) \
            .dense(int(m / 2)) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder

    def build_duplicate_pic(self) -> ModelBuilder:
        flow = self.flows_train[DataTypes.DUPLICATE_PIC.value]
        m = self.func()(flow.shape[1])
        builder = ModelBuilder() \
            .input(shape=(flow.shape[1]), name=DataTypes.DUPLICATE_PIC.value) \
            .dense(m, activation='relu') \
            .dropout(0.2) \
            .dense(int(m / 2)) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .softmax()
        self.inputs.append(builder.get_input())
        return builder 

    # def build_graph(self) -> ModelBuilder:
    #     flow = self.flows_train[DataTypes.GRAPH.value]
    #     m = DataHelper.max_pow_2(flow.shape[1])
    #     builder = ModelBuilder() \
    #         .input(shape=(flow.shape[1]), name=DataTypes.GRAPH.value) \
    #         .dense(32, activation='relu') \
    #         .dropout(0.2) \
    #         .dense(self.class_cnt) \
    #         .dense(self.class_cnt) \
    #         .softmax()
    #     self.inputs.append(builder.get_input())
    #     return builder
    
    def build_model(self) -> Model5: 
        self.builders = {}
        self.inputs = []
        self.builders[DataTypes.IMAGE.value] = self.build_image()
        self.builders[DataTypes.TEXT.value] = self.build_text()
        self.builders[DataTypes.CATEGORICAL.value] = self.build_categorical()
        self.builders[DataTypes.NUMERICAL.value] = self.build_numerical()
        self.builders[DataTypes.DUPLICATE_PIC.value] = self.build_duplicate_pic()
        # self.builders[DataTypes.GRAPH.value] = self.build_graph()

        self.builder = ModelBuilder() \
            .avg([builder.get_output() for _, builder in self.builders.items()]) \
            .dense(self.class_cnt) \
            .dense(self.class_cnt) \
            .set_optimizer(tf.keras.optimizers.Adam(learning_rate=1e-3)) \
            .set_loss_fn(tf.keras.losses.CategoricalCrossentropy(from_logits=True)) \
            .set_metric(tf.keras.metrics.AUC(from_logits=True, multi_label=True, num_labels=self.class_cnt))
        return self

    def train_model(self) -> Model5: 
        self.builder \
        .build_model(inputs=self.inputs) \
        .plot()
        self.builder.train(
            self.flows_train, 
            self.labels[:self.train_count],
            self.flows_validation,
            self.labels[self.train_count:],
            Model5.BATCH_SIZE,
            Model5.EPOCHS,
        )
        self.builder.plotlib()
        return self

    def evaluate_model(self) -> Model5:
        self.builder \
        .load_model(epoch=self.builder.best_epoch) \
        .get_model() \
        .evaluate(
            self.flows_validation,
            self.labels[self.train_count:],
            batch_size=Model5.BATCH_SIZE,
            verbose=2,
        )
        # result = self.builder.get_model().predict(
        #     self.flows_validation,
        #     batch_size=Model5.BATCH_SIZE,
        #     verbose=2,
        # )
        # metric = tf.keras.metrics.AUC(from_logits=True)
        # metric.update_state(self.labels[-1, :], np.vectorize(lambda x: 1 - x)(result[-1, :]))
        # print('%f' % metric.result())
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        x.reset_index(drop=True, inplace=True)
        x_flow = self.pack_flows(x)
        prediction = self.builder \
        .softmax() \
        .build_model(inputs=self.inputs) \
        .get_model() \
        .predict(x_flow, batch_size=32, verbose=2)

        prediction = np.vectorize(lambda x: 1 - x)(prediction[:, 0])
        
        return pd.DataFrame(
            data={
                'prediction': prediction,
            }
        )

    def get_model(self) -> Model5:
        return self.model
