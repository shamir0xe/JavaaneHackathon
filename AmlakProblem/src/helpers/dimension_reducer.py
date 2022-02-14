from __future__ import annotations
import pandas as pd
import sklearn
from src.models.reducer_types import ReducerTypes
# from src.helpers.config_reader import ConfigReader
from sklearn.decomposition import PCA


class DimensionReducer:
    VAR_THRESHOLD = 0.95

    def __init__(
        self, 
        data: pd.DataFrame,
        reducer_type: str = ReducerTypes.PCA
    ) -> None:
        self.data = data.to_numpy()
        self.reducer_type = reducer_type
    
    def optimal_components(self) -> DimensionReducer:
        # self.components_count = ConfigReader.read('dimension_reducer.components_count_default')
        self.components_count = 200
        self.run()
        variances = self.reducer.explained_variance_ratio_
        s, index = 0, 0
        for var in variances:
            s += var
            index += 1
            if s > DimensionReducer.VAR_THRESHOLD:
                self.components_count = index
                print(f'NUMBER OF COMPONENTS = {index}')
                break
        return self

    def run(self) -> DimensionReducer:
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
            res.append(f"{self.reducer_type.name}_{x}")
        return res

    def __pca(self) -> None:
        self.reducer = PCA(
            n_components=self.components_count,
            # svd_solver=ConfigReader.read('dimension_reducer.svd_solver'),
            svd_solver="randomized",
            # whiten=ConfigReader.read('dimension_reducer.whiten')
            whiten=True
        ).fit(self.data)
