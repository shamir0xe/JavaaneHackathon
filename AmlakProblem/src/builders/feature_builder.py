from __future__ import annotations
import pandas as pd
from src.models.distance_types import DistanceTypes
from src.models.normalize_types import NormalizeTypes
from src.helpers.config_reader import ConfigReader
from src.helpers.data_helper import DataHelper
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
    
    def get_extend_data(self, exception_list: list=[], selection_list: list=[]) -> pd.DataFrame:
        if len(selection_list) == 0:
            selection_list = self.data_output.columns
        else:
            selection_list = [*selection_list, 'token']
        selection_list = list(set(selection_list) - set(exception_list))
        not_available_list = list(set(selection_list) - set(self.data_output.columns))
        selection_list = list(set(selection_list) - set(not_available_list))
        selection_list.sort()
        return self.data_output[selection_list]

    def concat_frames(self, frame: pd.DataFrame) -> None:
        self.data_output = pd.concat([self.data_output, frame], axis=1)

    def graph_builder_factory(self, distance_type: str) -> GraphBuilder:
        # reading and normalizing data first
        data = DataHelper.normalize(self.data[ConfigReader.read('features.similarity.numerical_selection')], mode=NormalizeTypes.STD)
        for category in ConfigReader.read('features.similarity.categorical_selection'):
            data = pd.concat([data, DataHelper.categorical_to_numerical(self.data, category)], axis=1)
        # generating graph builder
        builder = GraphBuilder(
            data=data,
            distance_type=distance_type
        ) \
            .normalize_data() \
            .cluster_data() \
            .build_graphs() \
            .build_adjacency() \
            .normalize_weights()
        return builder

    def feature_aimilarity(self) -> None:
        # adding similarity feature
        for distance_type in [DistanceTypes.ACOS, DistanceTypes.EUCLIDIAN]:
            # creating builder
            builder = self.graph_builder_factory(distance_type=distance_type)
            # adding centralities
            self.concat_frames(builder.get_degree_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))
            self.concat_frames(builder.get_mean_weights())
            self.concat_frames(builder.get_load_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))

            # self.concat_frames(builder.get_harmonic_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))
            # self.concat_frames(builder.get_closeness_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))
            # self.concat_frames(builder.get_betweenness_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))

            # try:
            #     self.concat_frames(builder.get_subgraph_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))
            # except BaseException as err :
            #     print('subgraph centrality error occured: ', err)
            # try:
            #     self.concat_frames(builder.get_second_order_centrality(threshold=ConfigReader.read('features.similarity.degree_centrality.threshold')))
            # except BaseException as err:
            #     print('second order centrality error occured: ', err)

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

    def feature_rent_type(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_type'
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

    def feature_rent_sale(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'rent_sale'
            )
        )

    def feature_chat_enabled(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'chat_enabled'
            )
        )

    def feature_parking(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'parking'
            )
        )

    def feature_elevator(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'elevator'
            )
        )

    def feature_user_type(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'user_type'
            )
        )

    def feature_warehouse(self) -> None:
        self.concat_frames(
            DataHelper.binary_encode_categories(
                self.data, 'warehouse'
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

