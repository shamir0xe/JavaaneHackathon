from __future__ import annotations
from email.mime import image
from operator import index
import re
import sys
from tkinter import W
from urllib.request import DataHandler
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from src.facades.terminal_arguments import TerminalArguments
from src.helpers.data_helper import DataHelper
from src.helpers.dimension_reducer import DimensionReducer
from src.models.reducer_types import ReducerTypes
from src.models.ml_models.model0 import Model0
from src.models.ml_models.model1 import Model1
from src.models.ml_models.model2 import Model2
from src.models.ml_models.model3 import Model3
from src.models.ml_models.model4 import Model4
from src.models.ml_models.model5 import Model5
from src.models.ml_models.model6 import Model6
from src.models.normalize_types import NormalizeTypes
from src.models.data_types import DataTypes
from src.builders.feature_builder import FeatureBuilder
from src.facades.config_reader import ConfigReader
from src.helpers.memory_helper import MemoryHelper
from src.helpers.time_helper import TimeHelper
from src.models.fill_metrics import FillMetrics


class AmlakDelegator:
    def __init__(self) -> None:
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.mode.chained_assignment = None
        np.set_printoptions(threshold=sys.maxsize)
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    def read_data(self) -> AmlakDelegator:
        # reading data
        self.whole_data = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.whole_data.name')))
        self.data_train = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.train.name')))
        self.data_train = DataHelper.fillna_numerical(self.data_train, FillMetrics.MEDIAN)
        self.data_train = DataHelper.fillna_categorical(self.data_train, FillMetrics.MOD)
        self.data_validation = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.validation.name')))
        self.data_validation = DataHelper.fillna_numerical(self.data_validation, FillMetrics.MEDIAN)
        self.data_validation = DataHelper.fillna_categorical(self.data_validation, FillMetrics.MOD)
        self.dataset = pd.concat([self.data_train, self.data_validation], axis=0)
        self.dataset.reset_index(drop=True, inplace=True)
        return self
    
    def append_diff_time(self) -> AmlakDelegator:
        self.dataset['diff_time'] = -self.dataset['first_published_at'] - self.dataset['call_date']
        return self
    
    def backup_data(self) -> AmlakDelegator:
        # just remember tokens
        self.dataset_bc = self.dataset.copy()
        self.dataset = self.dataset[['token']]
        return self
    
    def append_bc_data(self) -> AmlakDelegator:
        # append numerical backups again
        self.dataset = DataHelper.left_join(self.dataset_bc, self.dataset, 'token')
        # self.data_validation = DataHelper.left_join(self.data_validation_bc, self.data_validation, 'token')
        return self

    def test(self) -> None:
        image_csv = pd.read_csv(DataHelper.data_path('pictures_data.csv'))
        print(image_csv[:10])
        image_csv = DataHelper.fillna_numerical(image_csv, FillMetrics.MEDIAN)
        print(image_csv[:10])
        self.dataset = image_csv
        self.dimension_reduction(op_name='color', prefer_component_cnt=1100)
        image_csv = self.dataset

        whole_data = pd.read_csv(DataHelper.data_path('0xwholedata.csv'))
        data_train = pd.read_csv(DataHelper.data_path('0xdata_train.csv'))
        data_validation = pd.read_csv(DataHelper.data_path('0xdata_validation.csv'))

        col_names = DataHelper.get_col_names(data_train, DataTypes.IMAGE)
        col_names = [col for col in col_names if re.match(r"(c_\d+_\d+_\d+)+", col)]
        data_train.drop(col_names, inplace=True, axis=1)
        data_train = DataHelper.left_join(data_train, image_csv, 'token')

        col_names = DataHelper.get_col_names(data_validation, DataTypes.IMAGE)
        col_names = [col for col in col_names if re.match(r"(c_\d+_\d+_\d+)+", col)]
        data_validation.drop(col_names, inplace=True, axis=1)
        data_validation = DataHelper.left_join(data_validation, image_csv, 'token')

        col_names = DataHelper.get_col_names(whole_data, DataTypes.IMAGE)
        col_names = [col for col in col_names if re.match(r"(c_\d+_\d+_\d+)+", col)]
        whole_data.drop(col_names, inplace=True, axis=1)
        whole_data = DataHelper.left_join(whole_data, image_csv, 'token')

        data_train.to_csv(DataHelper.data_path('0x0xdata_train.csv'), index=False)
        data_validation.to_csv(DataHelper.data_path('0x0xdata_validation.csv'), index=False)
        whole_data.to_csv(DataHelper.data_path('0x0xwhole_data.csv'), index=False)

        # whole_data = DataHelper.left_join(whole_data, tag_data, 'token')
        # data_train = DataHelper.left_join(data_train, tag_data, 'token')
        # data_validation = DataHelper.left_join(data_validation, tag_data, 'token')
        # whole_data.to_csv(DataHelper.data_path('0xwholedata.csv'), index=False)
        # data_train.to_csv(DataHelper.data_path('0xdata_train.csv'), index=False)
        # data_validation.to_csv(DataHelper.data_path('0xdata_validation.csv'), index=False)
    
    def test_2(self) -> AmlakDelegator:
        data_train = pd.read_csv(DataHelper.data_path('0x0xdata_train.csv'))
        data_validation = pd.read_csv(DataHelper.data_path('0x0xdata_validation.csv'))

        text_1 = data_train[[*DataHelper.get_col_names(data_train, DataTypes.TEXT), 'token']]
        text_2 = data_validation[[*DataHelper.get_col_names(data_validation, DataTypes.TEXT), 'token']]

        text = pd.concat([text_1, text_2], axis=0)
        text.reset_index(inplace=True, drop=True)

        text = DataHelper.fillna_numerical(text, FillMetrics.MEDIAN)
        self.dataset = text
        self.dimension_reduction(op_name='text')
        text = self.dataset

        col_names = DataHelper.get_col_names(data_train, DataTypes.TEXT)
        data_train.drop(col_names, inplace=True, axis=1)
        data_train = DataHelper.left_join(data_train, text, 'token')

        col_names = DataHelper.get_col_names(data_validation, DataTypes.TEXT)
        data_validation.drop(col_names, inplace=True, axis=1)
        data_validation = DataHelper.left_join(data_validation, text, 'token')

        data_train.to_csv(DataHelper.data_path('0x0x0xdata_train.csv'), index=False)
        data_validation.to_csv(DataHelper.data_path('0x0x0xdata_validation.csv'), index=False)

        return self


    def modify_labels(self) -> AmlakDelegator:
        # generating correct label for training
        self.labels_cnt = 3
        dataframe = pd.DataFrame(np.zeros((self.data_train.shape[0], self.labels_cnt)), columns=[f'l[{index}]' for index in range(self.labels_cnt)])
        real_cats = 0
        for i in range(dataframe.shape[0]):
            result = self.results.iloc[i]
            cat = round(self.cats.iloc[i]) + 1
            tag = self.tags.iloc[i]
            real_cats = max(cat + 1, real_cats)
            if result == 'ok':
                # ok posts
                dataframe.iloc[i, 0] = 1
            # else:
                # dataframe.iloc[i, min(cat, 6)] = 1
            elif not pd.isna(tag) and 'عکس' in tag:
                # inappropriate picture
                dataframe.iloc[i, 1] = 1
            # elif not pd.isna(tag) and int(self.dataset['categorical_duplicate_pic_1'].iloc[i]) == 1:
                # duplicate picture
                # dataframe.iloc[i, 2] = 1
            else:
                dataframe.iloc[i, 2] = 1
            # elif not pd.isna(tag) and 'عکس' in tag:
            #     # reason is pic
            #     dataframe.iloc[i, 1] = 1
            # else:
                # other reasons
                # dataframe.iloc[i, 2] = 1
        # for i in range(50):
        #     print('%s, %s, %s' % (dataframe.iloc[i, :], self.data_train['result'].iloc[i], self.data_train['tag'].iloc[i]))
        # exit()
        self.labels = dataframe
        # self.labels_cnt = real_cats
        return self

    def select_columns(self) -> AmlakDelegator:
        # selecting numerical columns
        dataframe = self.dataset[ConfigReader.read('selection_list')]
        dataframe = dataframe.add_prefix('num_')

        # adding token
        selection_list = ['token']
        # adding text vars
        selection_list = [*selection_list, *DataHelper.get_col_names(self.data_train, DataTypes.TEXT)]

        # adding pic vars
        selection_list = [*selection_list, *DataHelper.get_col_names(self.data_train, DataTypes.IMAGE)]

        dataframe = pd.concat([dataframe, self.dataset[selection_list]], axis=1)
        dataframe.reset_index(drop=True, inplace=True)

        # adding publish date
        dataframe = pd.concat([dataframe, TimeHelper.pd_to_days(self.dataset['first_published_at'])], axis=1)
        dataframe.reset_index(drop=True, inplace=True)

        # adding call date
        dataframe = pd.concat([dataframe, TimeHelper.pd_to_days(self.dataset['call_date'])], axis=1)
        dataframe.reset_index(drop=True, inplace=True)

        self.tags = self.dataset['tag']
        self.results = self.dataset['result']
        self.cats = self.dataset['tag_num']
        self.labels_cnt = len(self.dataset['tag_num'].unique()) + 1

        self.dataset = dataframe

        # print(self.dataset[:5])
        return self
    
    def normalize_dataset(self, mode: str=NormalizeTypes.BOTH) -> AmlakDelegator:
        self.dataset = DataHelper.normalize(self.dataset, except_cols=['token'], mode=mode)
        # numerical_frames = DataHelper.normalize(self.dataset[DataHelper.get_col_names(self.dataset, DataTypes.NUMERICAL)], mode=NormalizeTypes.BOTH)
        # self.dataset.update(numerical_frames)
        # self.dataset.reset_index(drop=True, inplace=True)
        # self.data_validation = DataHelper.normalize(self.data_validation, except_cols=['token'], mode=mode)
        return self

    def normalize_data(self) -> AmlakDelegator:
        # filling na values
        self.dataset = DataHelper.fillna_numerical(self.dataset, FillMetrics.MEDIAN)
        # normalizing data
        self.normalize_dataset()
        return self

    def ml_procedure(self) -> AmlakDelegator:
        print(self.dataset[:10])
        # print(self.labels[:10])
        self.model = Model6(
            data=self.dataset.drop(['token'], axis=1)[:self.data_train.shape[0]],
            labels=self.labels,
            class_cnt=self.labels_cnt
        ) \
        .shuffle_data() \
        .data_preparation() \
        .build_model() \
        .train_model() \
        .evaluate_model()
        return self

    def join_dataset(self, data: pd.DataFrame, on: str) -> None:
        self.dataset = DataHelper.left_join(self.dataset, data, on)
        # self.data_validation = DataHelper.left_join(self.data_validation, data, on)

    def build_pic_graph(self) -> nx.Graph:
        graph = nx.Graph()
        data = pd.read_csv(DataHelper.data_path('image_graph.csv'))
        for _, row in data.iterrows():
            u = row['token']
            for j in range(9):
                v = row[f'n_{j}']
                if pd.isna(v):
                    continue
                print('added edge %s->%s' % (u, v))
                graph.add_edge(u, v)
        return graph

    def append_pic_graph(self) -> AmlakDelegator:
        if TerminalArguments.exists('pic_cache') and MemoryHelper.cached('pic_cache'):
            pic_dataframe = MemoryHelper.retrieve('pic_cache')[0]
            self.dataset = pd.concat([self.dataset, pic_dataframe], axis=1)
            self.dataset.reset_index(drop=True, inplace=True)
            return self
        col = 'first_published_at'
        self.whole_data.update(TimeHelper.pd_to_days(self.whole_data[col]))
        self.whole_data.reset_index(drop=True, inplace=True)
        mapped_index = {}
        res = np.full(shape=(self.dataset.shape[0], 1), fill_value="0", dtype=object)
        graph = self.build_pic_graph()

        for u in tqdm(graph.nodes()):
            for index, row in self.dataset.iterrows():
                if row['token'] == u:
                    mapped_index[u] = index
                    break

        for u, v in tqdm(graph.edges()):
            time_u = int(self.whole_data[self.whole_data.token.eq(u)][col])
            time_v = int(self.whole_data[self.whole_data.token.eq(v)][col])
            # print('%d /// %d' % (time_u, time_v))
            if time_u < time_v and v in mapped_index:
                index = mapped_index[v]
                # print('INDEX of %s: %d' % (v, index))
                res[index] = "1"
            elif time_u > time_v and u in mapped_index:
                index = mapped_index[u]
                # print('INDEX of %s: %d' % (u, index))
                res[index] = "1"
        
        pic_dataframe = pd.DataFrame(res, columns=['duplicate_pic'])
        pic_dataframe = DataHelper.binary_encode_categories(pic_dataframe, 'duplicate_pic')
        self.dataset = pd.concat([self.dataset, pic_dataframe], axis=1)
        self.dataset.reset_index(drop=True, inplace=True)
        MemoryHelper.save('pic_cache', (pic_dataframe, ))
        return self

    def append_features(self) -> AmlakDelegator:
        # append features from whole data
        if TerminalArguments.exists('cache') and MemoryHelper.cached('categories_dataset'):
            self.dataset = MemoryHelper.retrieve('categories_dataset')[0]
            return self
        whole_data = pd.read_csv(DataHelper.data_path(ConfigReader.read('data.whole_data.name')), low_memory=False)
        whole_data = DataHelper.fillna_numerical(whole_data, FillMetrics.MEDIAN)
        whole_data = DataHelper.fillna_categorical(whole_data, FillMetrics.MOD)
        # print(whole_data[:10])
        builder = FeatureBuilder(
            data=whole_data,
        ) \
        .extract_token() \
        .apply_features()

        self.join_dataset(builder.get_extend_data(exception_list=ConfigReader.read('features.no_dim_reduction_list')), 'token')
        self.dimension_reduction(op_name='categorical')
        self.join_dataset(builder.get_extend_data(selection_list=ConfigReader.read('features.no_dim_reduction_list')), 'token')

        print(self.dataset[:10])
        self.dataset = DataHelper.fillna_numerical(self.dataset, FillMetrics.MEDIAN)
        # self.data_validation = DataHelper.fillna_numerical(self.data_validation, FillMetrics.MEDIAN)
        # print(self.data_train[:10])
        # self.join_shit()
        # normalizing data
        # self.normalize_dataset(mode=NormalizeTypes.MAX_MIN)
        MemoryHelper.save('categories_dataset', (self.dataset, ))
        return self
    
    def dimension_reduction(self, op_name: str='categorical', prefer_component_cnt: int=350) -> AmlakDelegator:
        # reducing dimensions
        reducer = DimensionReducer(
            data=self.dataset.drop(['token'], axis=1), 
            operation_name=op_name,
            component_cnt=prefer_component_cnt,
            reducer_type=ReducerTypes.PCA
        ) \
            .optimal_components() \
            .run()
        dataframe = reducer.transform(self.dataset.drop(['token'], axis=1))
        self.dataset = pd.concat([dataframe, self.dataset['token']], axis=1)
        self.dataset.reset_index(drop=True, inplace=True)
        return self
    
    def predict_result(self) -> AmlakDelegator:
        # predicting result
        self.response = self.model.predict(self.dataset.drop(['token'], axis=1)[self.data_train.shape[0]:])
        return self
    
    def edit_columns(self) -> AmlakDelegator:
        var_cols = [col for col in self.dataset.columns if 'var' in col]
        color_cols = [col for col in self.dataset.columns if re.match(r"((c_\d+_\d+_\d+)+|(var_\d)+)", col)]
        text_cols = [col for col in self.dataset.columns if re.match(r"text_pca_\d+", col)]
        # self.dataset = self.dataset.drop(var_cols, axis=1)
        # self.dataset = self.dataset.drop(color_cols, axis=1)
        # self.dataset = self.dataset.drop(text_cols, axis=1)
        return self

    def output(self) -> None:
        # saving tesult to the output file
        self.response['token'] = self.data_validation['token']
        self.response.to_csv(DataHelper.data_path(ConfigReader.read('output.name')), index=False)
    
    def dummy(self) -> None:
        data_train = pd.read_csv(DataHelper.data_path('data_train_ptext.csv'))
        data_validation = pd.read_csv(DataHelper.data_path('data_validation_ptext.csv'))
        data_with_pics_train = pd.read_csv(DataHelper.data_path('data_train_pic.csv'))
        data_with_pics_validation = pd.read_csv(DataHelper.data_path('data_validation_pic.csv'))

        pic_cols = [col for col in data_with_pics_train.columns if re.match(r"((c_\d+_\d+_\d+)+|(var_\d)+)", col)]
        pic_cols = [*pic_cols, 'token']
        data_train = DataHelper.left_join(data_train, data_with_pics_train[pic_cols], 'token')
        data_validation = DataHelper.left_join(data_validation, data_with_pics_validation[pic_cols], 'token')

        data_train.to_csv(DataHelper.data_path('final_data_train.csv'), index=False)
        data_validation.to_csv(DataHelper.data_path('final_data_validation.csv'), index=False)
