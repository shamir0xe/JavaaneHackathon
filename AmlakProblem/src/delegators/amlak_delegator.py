from __future__ import annotations
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
from src.helpers.data_helper import DataHelper
from src.helpers.dimension_reducer import DimensionReducer
from src.models.reducer_types import ReducerTypes
from src.models.ml_models.model0 import Model0
from src.models.ml_models.model1 import Model1
from src.models.ml_models.model2 import Model2
from src.builders.feature_builder import FeatureBuilder

class AmlakDelegator:
    def __init__(self) -> None:
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.mode.chained_assignment = None
        np.set_printoptions(threshold=sys.maxsize)

    def read_data(self) -> AmlakDelegator:
        # self.data = pd.read_parquet(DataHelper.data_path('posts.parquet'), 'pyarrow')
        self.data_train = pd.read_csv(DataHelper.data_path('amlakTrain24_2.csv'))
        self.data_validation = pd.read_csv(DataHelper.data_path('amlakValidation24_2.csv'))
        self.whole_data = pd.read_csv(DataHelper.data_path('amlak_all_242200.csv'), low_memory=False)
        # print(self.whole_data.columns)
        # print(self.data_train[:33])
        # print(self.whole_data[:10])
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
        selection_list = ['token', 'image_count', 'ladder_count', 'click_bookmark', 'click_contact', 'click_post', 'size', 'price_value', 'floor']
        self.data_train = self.data_train[selection_list]
        self.data_validation = self.data_validation[selection_list]
        return self
    
    def normalize_data(self) -> AmlakDelegator:
        self.data_train = DataHelper.normalize(self.data_train, ['token'])
        self.data_validation = DataHelper.normalize(self.data_validation, ['token'])

        # correcting labels
        self.labels_train["result"].replace({
            "ok": 0,
            "fake-post": 1
        }, inplace=True)
        return self

    def ml_procedure(self) -> AmlakDelegator:
        self.model = Model2(
            data=self.data_train.drop(['token'], axis=1),
            labels=self.labels_train
        ) \
        .data_preparation() \
        .build_model() \
        .train_model() \
        .evaluate_model()
        return self

    def append_features(self) -> AmlakDelegator:
        data_extend = FeatureBuilder(
            data=self.whole_data, 
        ) \
        .extract_token() \
        .apply_features() \
        .get_extend_data()
        self.data_train = DataHelper.inner_join(self.data_train, data_extend, 'token')
        self.data_validation = DataHelper.inner_join(self.data_validation, data_extend, 'token')
        return self

    def dimension_reduction(self) -> AmlakDelegator:
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
        self.response = self.model.predict(self.data_validation.drop(['token'], axis=1))
        return self

    def output(self) -> None:
        self.response['token'] = self.data_validation['token']
        self.response.to_csv(DataHelper.data_path('output.csv'), index=False)
