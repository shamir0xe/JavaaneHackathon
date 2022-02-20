from __future__ import annotations
from operator import index
from typing import Text
import pandas as pd
import numpy as np
from src.helpers.dimension_reducer import DimensionReducer
from src.models.reducer_types import ReducerTypes
from src.builders.data_generator import DataGenerator
from src.helpers.data_helper import DataHelper
from src.facades.config_reader import ConfigReader
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


class TextDelegator:
    def set_config(self) -> TextDelegator:
        self.vector_size = ConfigReader.read('text.vector_size')
        self.whitelist = DataHelper.persian_whitelist() + DataHelper.space_chars() 
        self.cols = ConfigReader.read('text.selection_list')
        return self

    def read_data(self) -> TextDelegator:
        self.dataset = pd.read_csv(DataHelper.data_path(ConfigReader.read('text.whole_data.name')), low_memory=False)
        self.dataset.reset_index(drop=True, inplace=True)
        self.data_train = pd.read_csv(DataHelper.data_path(ConfigReader.read('text.train.name')))
        self.data_train = self.data_train.drop(['number'], axis=1)
        self.data_validation = pd.read_csv(DataHelper.data_path(ConfigReader.read('text.validation.name')))
        self.data_validation = self.data_validation.drop(['number'], axis=1)
        self.data = pd.concat([self.data_train, self.data_validation], axis=0)
        self.data.reset_index(drop=True, inplace=True)
        return self

    def select_columns(self) -> TextDelegator:
        # self.dataset = self.dataset['token']
        return self

    def dimension_reduction(self) -> TextDelegator:
        # reducing dimensions
        reducer = DimensionReducer(
            data=self.dataset.drop(['token'], axis=1), 
            operation_name='text',
            component_cnt=ConfigReader.read('text.final_col_count'),
            reducer_type=ReducerTypes.PCA
        ) \
            .optimal_components() \
            .run()
        self.dataset = pd.concat([
            reducer.transform(self.dataset.drop(['token'], axis=1)), 
            self.dataset['token']
        ], axis=1)
        self.dataset.reset_index(drop=True, inplace=True)
        return self

    def generate_database(self) -> TextDelegator:
        print('in generate database')
        database = []
        for col in self.cols:
            for index, doc in enumerate(tqdm(self.dataset[col])):
                database.append(
                    TaggedDocument(
                        DataGenerator.word_array(
                            doc, 
                            self.whitelist
                        ), 
                        f'{index}-{col}'
                    )
                )
        self.database = database
        return self

    def ml_procedure(self) -> TextDelegator:
        print('in ml procedure')
        self.model = Doc2Vec(
            self.database, 
            vector_size=self.vector_size, 
            window=3, 
            min_count=1, 
            workers=111
        )
        return self
    
    def backup_data(self) -> TextDelegator:
        # just remember nececery data
        self.dataset_bc = self.dataset.copy()
        self.dataset = self.dataset[[*ConfigReader.read('text.selection_list'), 'token']]
        return self
    
    def restore_data(self) -> TextDelegator:
        self.dataset = DataHelper.left_join(self.dataset_bc, self.dataset, 'token')
        return self

    def join_dataset(self) -> TextDelegator:
        print('joining dataset')
        # self.dataset = DataHelper.left_join(self.dataset, self.data, 'token')
        self.data = DataHelper.left_join(self.data, self.dataset, 'token')
        # self.data_validation = DataHelper.left_join(self.data_validation, self.data, 'token')
        return self

    def predict(self) -> TextDelegator:
        print('predicting result')
        for col in self.cols:
            self.dataset = self.append_prediction(
                self.dataset,
                col,
            )
        # print(self.data_train[:11])
        # print(self.data_validation[:11])
        return self

    def drop_redundants(self) -> TextDelegator:
        self.dataset = self.dataset.drop(labels=self.cols, axis=1)
        # self.data_validation = self.data_validation.drop(labels=self.cols, axis=1)
        return self

    def split_dataset(self) -> TextDelegator:
        # spliting data
        self.data_train = self.data[:self.data_train.shape[0]]
        self.data_validation = self.data[self.data_train.shape[0]:]
        self.data_validation.reset_index(drop=True, inplace=True)
        return self

    def output(self) -> TextDelegator:
        self.dataset.to_csv(DataHelper.data_path(ConfigReader.read('text.whole_data.output')), index=False)
        self.data_train.to_csv(DataHelper.data_path(ConfigReader.read('text.train.output')), index=False)
        self.data_validation.to_csv(DataHelper.data_path(ConfigReader.read('text.validation.output')), index=False)
        return self
     
    def append_prediction(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        res = np.empty(shape=(data.shape[0], self.vector_size), dtype=float)
        for index, doc in enumerate(tqdm(data[col])):
            res[index] = self.model.infer_vector(
                DataGenerator.word_array(
                    doc,    
                    self.whitelist
                )
            )
        res = pd.concat([
            data, 
            pd.DataFrame(
                res, 
                columns=[f'infer_{index}_{col}' for index in range(self.vector_size)]
            )
        ], axis=1)
        res.reset_index(drop=True, inplace=True)
        return res
    
    def dummy(self) -> TextDelegator:
        whole_data = pd.read_csv(DataHelper.data_path('amlak_all_261800.csv'), low_memory=False)
        description_2 = pd.read_csv(DataHelper.data_path('texts.csv'), low_memory=False)
        from_description = pd.read_csv(DataHelper.data_path('from_description.csv'), low_memory=False)
        
        whole_data = DataHelper.left_join(whole_data, description_2, 'token')
        whole_data = DataHelper.left_join(whole_data, from_description, 'token')
        whole_data.to_csv(DataHelper.data_path('whole_data.csv'), index=False)
        return self

