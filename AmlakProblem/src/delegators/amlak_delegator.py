from __future__ import annotations
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
from src.helpers.data_helper import DataHelper
from src.models.model0 import Model0
from src.models.model1 import Model1
from src.builders.feature_builder import FeatureBuilder

class AmlakDelegator:
    def __init__(self) -> None:
        pd.set_option("display.max_colwidth", None)
        pd.set_option('display.max_rows', None)
        pd.options.mode.chained_assignment = None
        np.set_printoptions(threshold=sys.maxsize)
        self.data_train = None

    def read_data(self) -> AmlakDelegator:
        # self.data = pd.read_parquet(DataHelper.data_path('posts.parquet'), 'pyarrow')
        self.data_train = pd.read_csv(DataHelper.data_path('amlakTrain.csv'))
        self.data_validation = pd.read_csv(DataHelper.data_path('amlakValidation.csv'))
        return self
    
    def trim_data(self) -> AmlakDelegator:
        indices = []
        for index, row in self.data_train.iterrows():
            # print(f'{index} -> {row["result"]}')
            if 'عکس' in str(row['tag']):
                indices.append(index)
        self.data_train.drop(index=indices, inplace=True)
        self.data_train.drop(columns=["number"], inplace=True)
        self.data_train.reset_index(drop=True, inplace=True)
        return self
    
    def select_columns(self) -> AmlakDelegator:
        self.labels_train = self.data_train[['result']]
        selection_list = ['image_count', 'ladder_count', 'click_bookmark', 'click_contact', 'click_post']
        self.data_train = self.data_train[selection_list]
        self.data_validation = self.data_validation[selection_list]
        return self
    
    def normalize_data(self) -> AmlakDelegator:
        # normalizing data
        self.data_train.fillna(0, inplace=True)
        self.data_train = (self.data_train - self.data_train.mean()) / self.data_train.std()
        self.data_train = (self.data_train - self.data_train.min()) / (self.data_train.max() - self.data_train.min())

        # correcting labels
        self.labels_train["result"].replace({
            "ok": 0,
            "fake-post": 1
        }, inplace=True)

        print(self.labels_train.head(10))
        print(self.data_train.head(10))
        return self

    def ml_procedure(self) -> AmlakDelegator:
        self.response = Model1(
            data=self.data_train,
            labels=self.labels_train
        ) \
        .data_preparation() \
        .build_model() \
        .train_model() \
        .evaluate_model() \
        .predict(self.data_validation.to_numpy(na_value=0.))
        return self

    def append_features(self) -> AmlakDelegator:
        builder = FeatureBuilder(
            data_train=self.data_train, 
            data_validation=self.data_validation
        ).apply_features()
        self.data_train = builder.get_train_data()
        self.data_validation = builder.get_validation_data()
        return self

    def output(self) -> None:
        # for index, row in self.data.iterrows():
        #     # if 'عکس' in row['result']:
        #     print(f'{index} -> {row["tag"]}')
        # num = 10
        # data['data'] = self.data.head(num)['data']
        # data['index'] = self.data.head(num).index
        # data['district'] = self.data.head(num)['district']
        # data['valid'] = self.data.head(num)['result']
        # print(self.data.head(10)['data', 'district'])
        # print(data)
        # print(self.data.head(num))
        # print(self.response)
        pass
