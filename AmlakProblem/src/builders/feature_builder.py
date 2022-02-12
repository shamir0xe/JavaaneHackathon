from __future__ import annotations
import pandas as pd
from src.builders.data_builder import DataBuilder


class FeatureBuilder:
    def __init__(
        self, 
        data_train: pd.DataFrame,
        data_validation: pd.DataFrame
    ) -> None:
        self.data_train = data_train
        self.data_validation = data_validation
        self.features = ['similarity']
    
    def apply_features(self) -> FeatureBuilder:
        for feature in self.features:
            getattr(FeatureBuilder, f"feature_{feature}")(self)
        return self

    def feature_similarity(self) -> None:
        # data = ...
        data = DataBuilder(pd.DataFrame([self.data_train, self.data_validation])) \
            .add_image_count() \
            .add_city() \
            .add_price() \
            .add_district() \
            .add_size() \
            .get_data()
        degrees = GraphBuilder(
            data=data,
            threshold=0.1
        ) \
            .calculate_distance() \
            .calculate_degree() \
            .get_degrees()
        degrees = pd.DataFrame([1 if x > 0 else 0 for x in degrees])
        self.data_train
        return degrees

    def feature_district(self) -> None:
        print('in district feature')

    def feature_duplicate_picture(self) -> None:
        print('in duplicate feature')

    def feature_incorrect_address(self) -> None:
        print('in incorrect address feature')

    def feature_rent_before(self) -> None:
        print('in rent before feature')

