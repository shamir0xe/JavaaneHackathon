import os
from typing import Any
from collections.abc import Iterable
import pandas as pd


class MemoryHelper:
    @staticmethod
    def database_path() -> str:
        return os.path.join(os.path.dirname(__file__), '..', '..', 'database', 'caches')

    @staticmethod
    def cached(filename: str) -> bool:
        bl = os.path.exists(os.path.join(MemoryHelper.database_path(), f'{filename}.cache'))
        bl = bl or os.path.exists(os.path.join(MemoryHelper.database_path(), f'{filename}_0.cache'))
        return bl
    
    @staticmethod
    def retrieve(filename: str) -> Any:
        if not MemoryHelper.cached(filename):
            return None
        if os.path.exists(os.path.join(MemoryHelper.database_path(), f'{filename}.cache')):
            return pd.read_json(os.path.join(MemoryHelper.database_path(), f'{filename}.cache'))
        index = 0
        res = []
        while MemoryHelper.cached(f'{filename}_{index}'):
            res.append(pd.read_json(os.path.join(MemoryHelper.database_path(), f'{filename}_{index}.cache')))
            index += 1
        return tuple(res)

    
    @staticmethod
    def save_dataframe(dataframe: pd.DataFrame, path: str) -> None:
        dataframe.to_json(path_or_buf=path)

    @staticmethod
    def save(filename: str, obj: Any) -> None:
        dirpath = MemoryHelper.database_path()
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if isinstance(obj, Iterable):
            for index, data in enumerate(obj):
                MemoryHelper.save_dataframe(data, os.path.join(dirpath, f'{filename}_{index}.cache'))
        else:
            MemoryHelper.save_dataframe(obj, os.path.join(dirpath, f'{filename}.cache'))

        