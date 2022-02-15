from __future__ import annotations
import sys
import pandas as pd
import numpy as np
from src.helpers.data_helper import DataHelper
from src.helpers.dimension_reducer import DimensionReducer
from src.models.reducer_types import ReducerTypes
from src.models.ml_models.model0 import Model0
from src.models.ml_models.model1 import Model1
from src.models.ml_models.model2 import Model2
from src.builders.feature_builder import FeatureBuilder
from src.helpers.config_reader import ConfigReader
from src.helpers.argument_helper import ArgumentHelper
from src.helpers.memory_helper import MemoryHelper



class AmlakDelegator:
    def __init__(self) -> None:
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.mode.chained_assignment = None
        np.set_printoptions(threshold=sys.maxsize)

    def read_data(self) -> AmlakDelegator:
        # reading data
        self.data_train = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.train.name')))
        self.data_validation = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.validation.name')))
        return self
    
    def trim_data(self) -> AmlakDelegator:
        # triming excessive data

        # indices = []
        # for index, row in self.data_train.iterrows():
        #     # print(f'{index} -> {row["result"]}')
        #     if 'عکس' in str(row['tag']):
        #         indices.append(index)
        # self.data_train.drop(index=indices, inplace=True)
        self.data_train.drop(columns=["number"], inplace=True)
        self.data_train.reset_index(drop=True, inplace=True)
        return self

    def backup_data(self) -> AmlakDelegator:
        # just remember tokens
        self.data_train_bc = self.data_train.copy()
        self.data_train = self.data_train[['token']]
        self.data_validation_bc = self.data_validation.copy()
        self.data_validation = self.data_validation[['token']]
        return self
    
    def append_numerical_data(self) -> AmlakDelegator:
        # append numerical backups again
        self.data_train = DataHelper.inner_join(self.data_train_bc, self.data_train, 'token')
        self.data_validation = DataHelper.inner_join(self.data_validation_bc, self.data_validation, 'token')
        return self
    
    def select_columns(self) -> AmlakDelegator:
        # select desired columns
        self.labels_train = self.data_train[['result']]
        selection_list = ConfigReader.read('selection_list')
        self.data_train = self.data_train[selection_list]
        self.data_validation = self.data_validation[selection_list]
        return self
    
    def normalize_data(self) -> AmlakDelegator:
        # normalizing data
        self.data_train = DataHelper.normalize(self.data_train, ['token'])
        self.data_validation = DataHelper.normalize(self.data_validation, ['token'])

        # correcting labels
        self.labels_train["result"].replace({
            "ok": 0,
            "fake-post": 1
        }, inplace=True)
        return self

    def ml_procedure(self) -> AmlakDelegator:
        # learning model
        self.model = Model2(
            data=self.data_train.drop(['token'], axis=1),
            labels=self.labels_train,
        ) \
        .data_preparation() \
        .build_model() \
        .train_model() \
        .evaluate_model()
        return self

    def append_features(self) -> AmlakDelegator:
        # append features from whole data
        if ArgumentHelper.exists('cache') and MemoryHelper.cached('categories_dataset'):
            self.data_train, self.data_validation = MemoryHelper.retrieve('categories_dataset')
            return self
        whole_data = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.whole_data.name')), low_memory=False)
        builder = FeatureBuilder(
            data=whole_data, 
        ) \
        .extract_token() \
        .apply_features()

        data_extend = builder.get_extend_data(exception_list=ConfigReader.read('features.no_dim_reduction_list'))
        self.data_train = DataHelper.inner_join(self.data_train, data_extend, 'token')
        self.data_validation = DataHelper.inner_join(self.data_validation, data_extend, 'token')
        self.dimension_reduction()

        # data_extend = builder.get_extend_data(selection_list=ConfigReader.read('features.no_dim_reduction_list'))
        # self.data_train = DataHelper.inner_join(self.data_train, data_extend, 'token')
        # self.data_validation = DataHelper.inner_join(self.data_validation, data_extend, 'token')

        # normalizing data
        self.data_train = DataHelper.normalize(self.data_train, ['token'])
        self.data_validation = DataHelper.normalize(self.data_validation, ['token'])
        MemoryHelper.save('categories_dataset', (self.data_train, self.data_validation))
        return self

    def dimension_reduction(self) -> AmlakDelegator:
        # reducing dimensions
        reducer = DimensionReducer(
            data=self.data_train.drop(['token'], axis=1), 
            reducer_type=ReducerTypes.PCA
        ) \
            .optimal_components() \
            .run()
        dataframe = reducer.transform(self.data_train.drop(['token'], axis=1))
        self.data_train = pd.concat([dataframe, self.data_train['token']], axis=1)
        dataframe = reducer.transform(self.data_validation.drop(['token'], axis=1))
        self.data_validation = pd.concat([dataframe, self.data_validation['token']], axis=1)
        return self

    def predict_result(self) -> AmlakDelegator:
        # predicting result
        self.response = self.model.predict(self.data_validation.drop(['token'], axis=1))
        return self

    def output(self) -> None:
        # saving tesult to the output file
        self.response['token'] = self.data_validation['token']
        self.response.to_csv(DataHelper.data_path(ConfigReader.read('output.name')), index=False)
