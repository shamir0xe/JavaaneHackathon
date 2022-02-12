import os


class DataHelper:
    @staticmethod
    def data_path(filename: str) -> str:
        return os.path.join(os.path.dirname(__file__), '../../database', filename)
