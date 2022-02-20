from asyncore import read
from datetime import datetime
import time
import pandas as pd


class TimeHelper:
    @staticmethod
    def pd_to_days(data: pd.Series) -> pd.Series:
        data = data.map(lambda readable_time: TimeHelper.str_to_days(readable_time, '%Y-%m-%d %H:%M:%S'))
        return data

    @staticmethod
    def str_to_days(readable_time: str, format_str: str) -> int:
        return round(TimeHelper.to_seconds(readable_time, format_str=format_str) / 24 / 60 / 60)
    
    @staticmethod
    def to_seconds(readable_time: str, format_str='%Y-%m-%d') -> int:
        # 2021-07-26 00:59:23
        try:
            d = datetime.strptime(readable_time, format_str)
        except BaseException as err:
            try:
                d = datetime.strptime(readable_time, '%Y-%m-%d')
            except BaseException:
                return 0.
        return round(time.mktime(d.timetuple()))
