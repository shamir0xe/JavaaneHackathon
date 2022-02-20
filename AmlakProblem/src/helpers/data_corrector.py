from __future__ import annotations
import re

from src.helpers.data_helper import DataHelper


class DataCorrector:
    def __init__(
        self, 
        data: str, 
        whitelist: str
    ) -> None:
        self.data = data
        self.whitelist = whitelist
    
    def remove_bad_chars(self) -> DataCorrector:
        self.data = re.sub('(<U\+....>)+', ' ', self.data)
        return self

    def convert_with_whitelist(self) -> DataCorrector:
        self.data = DataHelper.convert_with_whitelist(self.data, self.whitelist)
        return self
    
    def to_array(self) -> list:
        data_array = self.data.split(' ')
        return [data for data in data_array if data != '']
    
    def to_string(self) -> str:
        return self.data
