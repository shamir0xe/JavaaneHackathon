from __future__ import annotations
import pandas as pd


class DataBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_output = pd.DataFrame()
 
    def get_data(self) -> pd.DataFrame:
        return self.data_output

    def add_city(self) -> DataBuilder:
        return self

    def add_district(self) -> DataBuilder:
        return self

    def add_category(self) -> DataBuilder:
        return self

    def add_price(self) -> DataBuilder:
        return self

    def add_size(self) -> DataBuilder:
        return self

    def add_data(self) -> DataBuilder:
        return self
