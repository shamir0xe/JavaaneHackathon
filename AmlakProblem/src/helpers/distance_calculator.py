import math
from turtle import distance
import numpy as np
from src.models.distance_types import DistanceTypes


class DistanceCalculator:
    EPS = 1e-6

    @staticmethod
    def euclidian(data1: np.array, data2: np.array) -> float:
        return np.linalg.norm(data1 - data2)
    
    @staticmethod
    def cos(data1: np.array, data2: np.array) -> float:
        l1 = np.linalg.norm(data1)
        l2 = np.linalg.norm(data2)
        if l1 < DistanceCalculator.EPS or l2 < DistanceCalculator.EPS:
            return 0.
        unit1 = data1 / l1
        unit2 = data2 / l2
        return np.dot(unit1, unit2)

    @staticmethod
    def calculate(data1: np.array, data2: np.array, distance_type: str) -> float:
        if distance_type is DistanceTypes.EUCLIDIAN:
            return DistanceCalculator.euclidian(data1, data2)
        elif distance_type is DistanceTypes.ACOS:
            return DistanceCalculator.cos(data1, data2)
        return 0.
