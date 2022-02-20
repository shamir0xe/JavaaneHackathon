from __future__ import annotations
import pandas as pd
import sklearn
from src.models.reducer_types import ReducerTypes
from sklearn.decomposition import PCA
from src.facades.config_reader import ConfigReader


class DimensionReducer:
    VAR_THRESHOLD = 0.825

    def __init__(
        self, 
        data: pd.DataFrame,
        operation_name: str,
        component_cnt: int = ConfigReader.read('dimension_reducer.components_count'),
        reducer_type: str = ReducerTypes.PCA
    ) -> None:
        data.reset_index(drop=True, inplace=True)
        self.data = data.to_numpy()
        self.reducer_type = reducer_type
        self.operation_name = operation_name
        self.components_count = component_cnt
    
    def optimal_components(self) -> DimensionReducer:
        # self.components_count = ConfigReader.read('dimension_reducer.components_count_default')
        # print('shapes: ', self.data.shape)
        self.components_count = min(self.components_count, self.data.shape[0], self.data.shape[1])
        self.run()
        variances = self.reducer.explained_variance_ratio_
        s, index = 0, 0
        for var in variances:
            s += var
            index += 1
            if s > DimensionReducer.VAR_THRESHOLD:
                self.components_count = index
                break
        return self

    def run(self) -> DimensionReducer:
        print(f'NUMBER OF COMPONENTS = {self.components_count}')
        if self.reducer_type is ReducerTypes.PCA:
            self.__pca()
            # self.reducer = sklearn.decomposition.PCA()
        elif self.reducer_type is ReducerTypes.TSNE:
            pass
        return self
    
    def transform(self, data_train: pd.DataFrame) -> pd.DataFrame:
        dataframe = self.reducer.transform(data_train.to_numpy())
        return pd.DataFrame(dataframe, columns=self.get_names())

    def get_names(self) -> list:
        res = []
        for x in range(self.components_count):
            res.append(f"{self.operation_name}_{self.reducer_type.name.lower()}_{x}")
        return res

    def __pca(self) -> None:
        self.reducer = PCA(
            n_components=self.components_count,
            # svd_solver=ConfigReader.read('dimension_reducer.svd_solver'),
            svd_solver="randomized",
            # whiten=ConfigReader.read('dimension_reducer.whiten')
            whiten=True
        ).fit(self.data)
