import os
import pandas as pd
import numpy as np
from tqdm import tqdm


class DataHelper:
    @staticmethod
    def data_path(filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), '../../database', filename)
    
    @staticmethod
    def inner_join(data1: pd.DataFrame, data2: pd.DataFrame, col: str) -> pd.DataFrame:
        return pd.merge(data1, data2, on=col)

    @staticmethod
    def normalize(data: pd.DataFrame, except_cols: list) -> pd.DataFrame:
        backup_data = data[except_cols]
        data = data.drop(except_cols, axis=1)
        data = data.fillna(0)
        data = (data - data.mean()) / data.std()
        data = (data - data.min()) / (data.max() - data.min())
        data = pd.concat([data, backup_data], axis=1)
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
        return pd.DataFrame([uniques[name_corrector(col)] for col in data[categorical_col]], columns=[categorical_col])
    
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
