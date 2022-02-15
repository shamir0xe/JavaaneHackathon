from __future__ import annotations
import pandas as pd
from src.helpers.config_reader import ConfigReader
from src.helpers.data_helper import DataHelper
from src.builders.data_builder import DataBuilder
from src.builders.graph_builder import GraphBuilder


class FeatureBuilder:
    def __init__(
        self, 
        data: pd.DataFrame,
    ) -> None:
        self.data = data
        self.data_output = pd.DataFrame()
    
    def apply_features(self) -> FeatureBuilder:
        for feature in self.get_features():
            print(f'building {feature}')
            getattr(FeatureBuilder, feature)(self)
        return self
    
    def extract_token(self) -> FeatureBuilder:
        self.data_output['token'] = self.data['token']
        return self

    def get_features(self) -> list:
        return [func for func in dir(FeatureBuilder) if \
            callable(getattr(FeatureBuilder, func)) and func.startswith('feature')
        ]
    
    def get_extend_data(self, exception_list: list = [], selection_list: list = []) -> pd.DataFrame:
        # TODO
        return self.data_output

    def concat_frames(self, frame: pd.DataFrame) -> None:
        self.data_output = pd.concat([self.data_output, frame], axis=1)

    def eature_similarity(self) -> None:
        # data = ...
        builder = GraphBuilder(
            numerical_data=self.data[ConfigReader.read('features.similarity.numerical_selection')],
            categorical_data=self.data[ConfigReader.read('features.similarity.categorical_selection')],
        ) \
            .cluster_data(cluster_count=ConfigReader.read('features.similarity.cluster_count')) \
            .build_adjacency() \
            .calculate_distances()
        # adding degrees
        self.concat_frames(builder.get_cummulative_distance())
        # degrees = pd.DataFrame([1 if x > 0 else 0 for x in degrees])
        # self.data_output['degrees'] = degrees
        # return degrees

    def feature_district(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'district'
            )
        ) 
    
    def feature_city(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'city'
            )
        )

    def feature_category(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'category'
            )
        )
    
    def feature_rent_sale(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_Sale'
            )
        )
    
    def feature_credit_mode(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'credit_mode'
            )
        )
    
    def feature_rent_mode(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_mode'
            )
        )

    def feature_price_mode(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'price_mode'
            )
        )

    def feature_parking(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'parking'
            )
        )

    def feature_chat(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'chat_enabled'
            )
        )

    def feature_room(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'room'
            )
        )
    
    def feature_elevator(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'elevator'
            )
        )

    def feature_rent_type(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_type'
            )
        )

    def feature_user_type(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'user_type'
            )
        )

    def feature_rent_to_single(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_to_single'
            )
        )

    def feature_rent_credit_transform(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_credit_transform'
            )
        )

    # def eature_duplicate_picture(self) -> None:
    #     pass
    #     # print('in duplicate feature')

    # def eature_incorrect_address(self) -> None:
    #     pass
    #     # print('in incorrect address feature')

    # def eature_rent_before(self) -> None:
    #     pass
    #     # print('in rent before feature')

