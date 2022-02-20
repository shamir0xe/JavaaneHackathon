from __future__ import annotations
from copyreg import pickle
from email.mime import base, image
import math
import sys
from urllib.request import DataHandler
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm
from sklearn import datasets
from src.helpers.distance_calculator import DistanceCalculator
from src.models.reducer_types import ReducerTypes
from src.models.fill_metrics import FillMetrics
from src.models.distance_types import DistanceTypes
from src.helpers.dimension_reducer import DimensionReducer
from src.helpers.data_corrector import DataCorrector
from src.helpers.data_helper import DataHelper
from src.helpers.picture_helper import PictureHelper
from src.facades.config_reader import ConfigReader


class PictureDelegator:
    def __init__(self) -> None:
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.options.mode.chained_assignment = None
        np.set_printoptions(threshold=sys.maxsize)

    def set_config(self) -> PictureDelegator:
        self.n_pictures = ConfigReader.read('picture.n_pictures')
        self.n_fraction = ConfigReader.read('picture.n_fraction')
        self.n_channel = 3
        return self

    def read_data(self) -> PictureDelegator:
        # self.data_train = pd.read_csv(DataHelper.data_path(ConfigReader.read('picture.train.name')))
        # self.data_validation = pd.read_csv(DataHelper.data_path(ConfigReader.read('picture.validation.name')))
        self.dataset = pd.read_csv(DataHelper.data_path('amlak_image.csv'), low_memory=False)
        self.dataset = self.dataset[self.dataset.is_image.eq(1)]
        self.dataset.reset_index(drop=True, inplace=True)
        return self
    
    def unify_dataset(self) -> PictureDelegator:
        self.dataset = pd.concat([self.data_train, self.data_validation], axis=0)
        self.dataset.reset_index(drop=True, inplace=True)
        return self
    
    def split_dataset(self) -> PictureDelegator:
        # spliting dataset into train/validation
        self.data_train = self.dataset.iloc[:self.data_train.shape[0], :]
        self.data_validation = self.dataset.iloc[self.data_train.shape[0]:, :]
        self.data_validation.reset_index(drop=True, inplace=True)
        return self
    
    def create_initial_arr(self, n_row: int) -> np.array:
        res = np.empty((n_row, self.n_pictures * self.n_channel * self.n_fraction ** 2))
        res[:, :] = np.NaN
        return res

    def has_image(self, index: int) -> bool:
        return int(self.dataset_bc['is_image'].iloc[index]) > 0

    def image_to_array(self) -> PictureDelegator:
        def col_extractor(index: int) -> str:
            res = 'c'
            # fraction
            res += f'_{index % self.n_fraction ** 2}'
            # channel
            index = int(index / self.n_fraction ** 2)
            res += f'_{index % self.n_channel}'
            # pic_idx
            index = int(index / self.n_channel)
            res += f'_{index}'
            return res
        n = self.dataset.shape[0]
        image_nums = np.zeros(shape=(n, 1), dtype=int)
        image_array = self.create_initial_arr(n)
        length = self.n_pictures * self.n_channel * self.n_fraction ** 2
        no_match = 0
        for i in tqdm(range(n)):
            if self.has_image(i):
                try:
                    post_list = PictureHelper.post_array(self.dataset['token'].iloc[i], n_picture=self.n_pictures, fraction=self.n_fraction)
                except BaseException as err:
                    no_match += 1
                    continue
                if len(post_list) == 0:
                    no_match += 1
                    continue
                temp = self.create_initial_arr(1)
                temp[0, :min(length, len(post_list))] = np.array(post_list)[:length]
                # print(temp.shape)
                image_array[i, :] = temp
                image_nums[i] = int(1e-9 + len(post_list) / 3 / self.n_fraction ** 2)
        # print([col_extractor(i) for i in range(length)])
        # print(self.dataset.shape)
        # print(image_array.shape)
        print('no matched count: ', no_match)
        self.dataset = pd.concat([self.dataset, pd.DataFrame(image_nums, columns=['image_count'])], axis=1)
        self.dataset = pd.concat([self.dataset, pd.DataFrame(image_array, columns=[col_extractor(i) for i in range(length)])], axis=1)
        self.dataset.reset_index(inplace=True, drop=True)
        return self

    def calculate_variances(self) -> PictureDelegator:
        n = self.dataset.shape[0]
        variances = np.empty((n, self.n_channel))
        variances[:, :] = np.NaN
        for i in tqdm(range(n)):
            if self.has_image(i):
                var = PictureHelper.image_variances(self.dataset['token'].iloc[i])
                if len(var) == 0:
                    continue
                variances[i] = np.array(var)
        self.dataset = pd.concat([self.dataset, pd.DataFrame(variances, columns=[f'var_{i}' for i in range(self.n_channel)])], axis=1)
        self.dataset.reset_index(inplace=True, drop=True)
        return self

    def backup_data(self) -> PictureDelegator:
        # just remember tokens
        self.dataset_bc = self.dataset.copy()
        self.dataset = self.dataset[['token']]
        return self

    def dimension_reduction(self) -> PictureDelegator:
        # reducing dimension
        self.dataset = DataHelper.fillna_numerical(self.dataset, FillMetrics.MEDIAN)
        reducer = DimensionReducer(
            data=self.dataset.drop(['token'], axis=1), 
            operation_name='picture',
            component_cnt=ConfigReader.read('picture.final_col_count'),
            reducer_type=ReducerTypes.PCA
        ) \
            .optimal_components() \
            .run()
        self.dataset = pd.concat([reducer.transform(self.dataset.drop(['token'], axis=1)), self.dataset['token']], axis=1)
        self.dataset.reset_index(inplace=True, drop=True)
        return self

    def restore_data(self) -> PictureDelegator:
        # restore dataset from backup data
        self.dataset = DataHelper.left_join(self.dataset_bc, self.dataset, 'token')
        return self

    def output(self) -> PictureDelegator:
        # self.data_train.to_csv(DataHelper.data_path(ConfigReader.read('picture.train.output')), index=False)
        # self.data_validation.to_csv(DataHelper.data_path(ConfigReader.read('picture.validation.output')), index=False)
        self.dataset.to_csv(DataHelper.data_path('pictures_data.csv'), index=False)
        return self

    def test(self) -> PictureDelegator:
        # id fake pic
        index = []
        count = 50
        for i in range(self.data_train.shape[0]):
            if self.data_train['result'][i] == 'ok' and int(self.data_train['image_count'][i]) > 0:
                index.append(i)
                count -= 1
                if count <= 0:
                    break
        for j, i in enumerate(index):
            print(self.data_train[['city', 'district', 'size', 'tag']].iloc[i,:])
            print(DataCorrector(data=self.data_train['description'][i], 
            whitelist=DataHelper.persian_whitelist() + DataHelper.space_chars() + DataHelper.english_numbers()) \
            .remove_bad_chars() \
            .convert_with_whitelist() \
            .to_string())
            for j in range(self.data_train['image_count'][i]):
                PictureHelper.open_image(
                    DataHelper.picture_path(self.data_train['token'][i], f'{j}.jpg')
                )
            input()
        return self

    def build_graph(self) -> PictureDelegator:
        data = pd.read_csv(DataHelper.data_path('pictures_data.csv'))
        arr = data.drop(['token', 'image_count'], axis=1)
        arr = arr.to_numpy()
        n = data.shape[0]
        base_length = self.n_channel * 3 ** 2
        m = base_length * self.n_pictures
        threshold = (((4.5 + 7) / 2 + 4.5) / 2 + (4.5 + 7) / 2) / 2 * math.sqrt(base_length)
        # var_th = (threshold / math.sqrt(base_length)) ** 2
        adj = [set() for _ in range(n)]
        # weights = [{} for _ in range(n)]
        max_adj = -1
        # max_distance = -1
        # best_index = 0
        for i in tqdm(range(n)):
            for ii in range(m):
                if np.isnan(arr[i, ii * base_length]):
                    break
                for j in range(n):
                    if j <= i:
                        continue
                    for jj in range(m):
                        if np.isnan(arr[j, jj * base_length]):
                            break
                        a, b = arr[i, ii * base_length: (ii + 1) * base_length],\
                            arr[j, jj * base_length: (jj + 1) * base_length]
                        distance = DistanceCalculator.calculate(a, b, DistanceTypes.EUCLIDIAN)
                        # variance = np.var(a - b)
                        if distance < threshold:
                            # if max_distance < distance:
                            #     max_distance = distance
                            #     best_index = (i, j)
                            adj[i].add(j)
                            # if j in weights[i]:
                            #     weights[i][j] = max(variance, weights[i][j])
                            #     weights[j][i] = weights[i][j]
                            # else:
                            #     weights[i][j] = variance
                            #     weights[j][i] = variance
                            adj[j].add(i)
                            max_adj = max(max_adj, len(adj[i]), len(adj[j]))
        # print('max distance', max_distance)
        # print('index: ', best_index)
        data = pd.read_csv(DataHelper.data_path('pictures_data.csv'))
        data = data[['token']]
        mat = np.empty(shape=(n, max_adj), dtype=object)
        for u in range(n):
            for index, v in enumerate(adj[u]):
                mat[u, index] = f"{data['token'].iloc[v]}"
        print(pd.DataFrame(mat, columns=[f'n_{index}' for index in range(max_adj)]), pd.DataFrame())
        data = pd.concat([data, pd.DataFrame(mat, columns=[f'n_{index}' for index in range(max_adj)])], axis=1)
        data.reset_index(drop=True, inplace=True)
        data.to_csv('image_graph.csv', index=False)
        return self