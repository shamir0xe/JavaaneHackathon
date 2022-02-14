from __future__ import annotations
import pandas as pd


class GraphBuilder:
    def __init__(
        self, 
        data: pd.DataFrame,
        baseline: pd.DataFrame,
    ) -> None:
        self.graph = []
        self.data = data
        self.baseline = baseline

    def calculate_distance(self) -> GraphBuilder:
        return self

    def calculate_degrees(self) -> GraphBuilder:
        return self

    def get_degrees(self) -> pd.DataFrame:
        return pd.DataFrame([len(x) for x in self.graph])
