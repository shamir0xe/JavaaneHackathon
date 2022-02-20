import math
import numbers
import os
import re
from unicodedata import numeric
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.facades.config_reader import ConfigReader
from libs.python_library.io.buffer_reader import BufferReader
from libs.python_library.io.file_buffer import FileBuffer
from src.models.normalize_types import NormalizeTypes
from src.models.fill_metrics import FillMetrics
from src.models.data_types import DataTypes


class DataHelper:
    @staticmethod
    def max_pow_2(number: int) -> int:
        i = 0
        while 2 ** i <= number:
            i += 1
        return 2 ** (i - 1)

    @staticmethod
    def binary_to_u16_string(bytes: list) -> str:
        value = 0
        for byte in bytes:
            value = value * 256 + byte
        res = ''
        while value > 0:
            res = str(value % 16) + res
            value /= 16
        res = f'<u+{res}>'
        return res.upper()

    @staticmethod
    def pack_flow(data: pd.DataFrame, data_type: str) -> np.array:
        columns = DataHelper.get_col_names(data, data_type)
        data = data[columns]
        return data.to_numpy(dtype="float32")

    @staticmethod
    def get_col_names(data: pd.DataFrame, category: str) -> list:
        if category is DataTypes.NUMERICAL:
            return [col for col in data.columns if re.match(r"(num_.*)", col)]
        if category is DataTypes.CATEGORICAL:
            return [col for col in data.columns if re.match(r"(categorical_.*)", col)]
        if category is DataTypes.TEXT:
            return [col for col in data.columns if re.match(r"(infer_\d+_.*)+|(text_.+)+", col)]
        if category is DataTypes.GRAPH:
            return []
        if category is DataTypes.IMAGE:
            return [col for col in data.columns if re.match(r"((c_\d+_\d+_\d+)+|(var_\d)+|(color_.+)+)", col)]
        if category is DataTypes.DUPLICATE_PIC:
            return [col for col in data.columns if re.match(r"(categorical_duplicate_pic_\d)+", col)]
        return []

    @staticmethod
    def persian_whitelist() -> str:
        return 'ابپتثجچحخدذزرزژسشصضطظعغفقکگلمنوهیئيآءٔهه ه'
    
    @staticmethod
    def space_chars() -> str:
        return ' '
    
    @staticmethod
    def english_numbers() -> str:
        return '0123456789'

    @staticmethod
    def convert_with_whitelist(doc: str, whitelist: str) -> str:
        doc = re.sub('ي', 'ی', doc)
        doc = re.sub('([\d]+)', ' \\1 ', doc)
        doc = re.sub('\s', ' ', doc)
        whitelist = list(whitelist)
        res = ''
        for char in doc:
            if char in whitelist:
                res += char
            else:
                res += ' '
        res = re.sub('  ', ' ', res)
        return res
    
    @staticmethod
    def bad_chars_bytearray() -> list:
        reader = BufferReader(FileBuffer(DataHelper.data_path('bad_chars')))
        res = []
        while not reader.end_of_buffer():
            line = reader.next_line().strip()
            res.append(bytearray(line.encode('utf-8')))
        return res

    @staticmethod
    def picture_path(*args) -> str:
        path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'JavaaneImages', 'images')
        for arg in args:
            path = os.path.join(path, str(arg))
        return path

    @staticmethod
    def data_path(filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), '../../database', filename)
    
    @staticmethod
    def cache_path(filename: str) -> str:
        if filename is None:
            return DataHelper.data_path('caches')
        return os.path.join(DataHelper.data_path('caches'), filename)
    
    @staticmethod
    def left_join(data1: pd.DataFrame, data2: pd.DataFrame, col: str) -> pd.DataFrame:
        return pd.merge(left=data1, right=data2, on=col, how='left')

    @staticmethod
    def normalize(data: pd.DataFrame, except_cols: list=[], mode: str=NormalizeTypes.BOTH) -> pd.DataFrame:
        def std(data: pd.DataFrame) -> pd.DataFrame:
            return (data - data.mean()) / data.std()

        def max_min(data: pd.DataFrame) -> pd.DataFrame:
            return (data - data.min()) / (data.max() - data.min())

        if len(except_cols) != 0:
            backup_data = data[except_cols]
            data = data.drop(except_cols, axis=1)
        # data = data.fillna(0)
        # data = DataHelper.fillna(data, FillMetrics.MEDIAN)
        if mode is NormalizeTypes.BOTH:
            data = max_min(std(data))
        elif mode is NormalizeTypes.STD:
            data = std(data)
        elif mode is NormalizeTypes.MAX_MIN:
            data = max_min(data)
        if len(except_cols) != 0:
            data = pd.concat([data, backup_data], axis=1)
        data.reset_index(drop=True, inplace=True)
        return data
    
    @staticmethod
    def categorical_cols(data: pd.DataFrame) -> list:
        res = []
        for index, dtype in data.dtypes.iteritems():
            if dtype in [object, str]:
                res.append(index)
        return res

    @staticmethod
    def numeric_cols(data: pd.DataFrame) -> list:
        return list(set(data.columns) - set(DataHelper.categorical_cols(data)))
        
    @staticmethod
    def fillna_numerical(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        numeric_cols = DataHelper.numeric_cols(data)
        if metric is FillMetrics.MEDIAN:
            temp = data[numeric_cols]
            temp.apply(pd.to_numeric, errors='coerce')
            temp = temp.median()
            jj = pd.DataFrame()
            for col in tqdm(numeric_cols):
                jj[col] = data[col].map(lambda x: 0 if pd.isna(temp[col]) else temp[col] if pd.isna(x) else x)
            data.update(jj)
        else:
            raise Exception('madareto')
        return data
    
    @staticmethod
    def fillna_categorical(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        return data
        categorical_cols = DataHelper.categorical_cols(data)
        if metric is FillMetrics.MOD:
            temp = data[categorical_cols]
            temp = temp.mode()
            print(temp)
            exit()
            jj = pd.DataFrame()
            for col in categorical_cols:
                jj[col] = data[col].map(lambda x: temp[col] if pd.isna(x) else x)
            data.update(jj)
        else:
            raise Exception('madareto')
        return data

    @staticmethod
    def fillna(data: pd.DataFrame, metric: str) -> pd.DataFrame:
        data.update(DataHelper.fillna_numerical(data, metric))
        data.update(DataHelper.fillna_categorical(data, metric))
        return data
    
    @staticmethod
    def binary_encode_categories(data: pd.DataFrame, categorical_col: str) -> pd.DataFrame :
        feature_col = DataHelper.categorical_to_numerical(data, categorical_col)
        binary_columns = DataHelper.binary_encode(feature_col)
        return binary_columns

    @staticmethod
    def categorical_to_numerical(data: pd.DataFrame, categorical_col: str) -> pd.DataFrame:
        def name_corrector(name: str) -> str:
            if name is None:
                name = 'unkown'
            if type(name) is not str:
                name = str(name)
            return name.strip()
        cols = data[categorical_col].unique()
        uniques = {}
        for col in cols:
            col = name_corrector(col)
            if not col in uniques:
                uniques[col] = len(uniques)
        return pd.DataFrame([uniques[name_corrector(col)] for col in data[categorical_col]], columns=[f'categorical_{categorical_col}'])
    
    @staticmethod
    def binary_encode(feature_col: pd.DataFrame) -> pd.DataFrame:
        # receving integer labels and returns 1hot encoding
        unique_count = feature_col.iloc[:, 0].max() + 1
        row_count = feature_col.shape[0]
        print(f'uniques, row_count = {unique_count}, {row_count}')
        res = pd.DataFrame(np.zeros((row_count, unique_count)), columns=[f'{feature_col.columns[0]}_{index}' for index in range(unique_count)])
        index = 0
        for number in tqdm(feature_col.iloc[:, 0], desc=feature_col.columns[0]):
            res.iloc[index, number] = 1
            index += 1
        return res
