from __future__ import annotations
import pandas as pd
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
            getattr(FeatureBuilder, feature)(self)
        return self
    
    def extract_token(self) -> FeatureBuilder:
        self.data_output['token'] = self.data['token']
        return self

    def get_features(self) -> list:
        return [func for func in dir(FeatureBuilder) if \
            callable(getattr(FeatureBuilder, func)) and func.startswith('feature')
        ]
    
    def get_extend_data(self) -> pd.DataFrame:
        return self.data_output

    def eature_similarity(self) -> None:
        # data = ...
        data = DataBuilder(self.data) \
            .add_image_count() \
            .add_city() \
            .add_district() \
            .add_category() \
            .add_price_mode() \
            .add_price() \
            .add_floor() \
            .add_size() \
            .normalize() \
            .get_data()
        degrees = GraphBuilder(
            data=data,
            threshold=0.1
        ) \
            .calculate_distance() \
            .calculate_degree() \
            .get_degrees()
        degrees = pd.DataFrame([1 if x > 0 else 0 for x in degrees])
        self.data_output['degrees'] = degrees
        return degrees

    def concat_frames(self, frame: pd.DataFrame) -> None:
        self.data_output = pd.concat([self.data_output, frame], axis=1)

    def feature_district(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_cols(
                self.data, 'district'
            )
        ) 
    
    def feature_city(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_cols(
                self.data, 'city'
            )
        )

    def feature_category(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_cols(
                self.data, 'category'
            )
        )
    
    def feature_floor(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_cols(
                self.data, 'floor'
            )
        )
    
    # def feature_rent_sale(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'rent_sale'
    #         )
    #     )
    
    # def feature_credit_mode(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'credit_mode'
    #         )
    #     )
    
    # def feature_rent_mode(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'rent_mode'
    #         )
    #     )

    # def feature_price_mode(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'price_mode'
    #         )
    #     )

    # def feature_parking(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'parking'
    #         )
    #     )

    # def feature_chat(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'chat_enabled'
    #         )
    #     )

    # def feature_room(self) -> None:
    #     self.concat_frames(
    #         DataHelper.binary_encode_cols(
    #             self.data, 'room'
    #         )
    #     )



    def eature_duplicate_picture(self) -> None:
        pass
        # print('in duplicate feature')

    def eature_incorrect_address(self) -> None:
        pass
        # print('in incorrect address feature')

    def eature_rent_before(self) -> None:
        pass
        # print('in rent before feature')

