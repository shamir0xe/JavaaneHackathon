from src.helpers.data_corrector import DataCorrector


class DataGenerator:
    @staticmethod
    def word_array(data: str, whitelist: str) -> list:
        return DataCorrector(data=data, whitelist=whitelist) \
            .remove_bad_chars() \
            .convert_with_whitelist() \
            .to_array()
