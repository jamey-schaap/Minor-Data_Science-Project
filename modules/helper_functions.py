import math

import numpy as np
import pandas as pd
from typing import Optional, Callable
from dataclasses import dataclass
from configs.data import A, B, Cols, GOV_INSTABILITY_LOOKBACK_YEARS, Prefs

@dataclass
class Row:
    country: str
    year: int
    gdp_pc: Optional[float]
    durable: Optional[int]

    @staticmethod
    def create(
            country: str,
            year: int,
            gdp_pc: float | None = None,
            durable: int | None = None):
        return Row(country, year, gdp_pc, durable)


def normalize(x: int | float, x_min: int | float, x_max: int | float) -> float:
    return A + (((x - x_min) * (B - A)) / (x_max - x_min))


def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = df

    column_min = np.nanmin(result_df[column_name])
    column_max = np.nanmax(result_df[column_name])
    result_df[f"{Prefs.NORM}{column_name}"] = [normalize(x, column_min, column_max) for x in result_df[column_name]]

    return result_df


def log_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = df
    result_df[f"{Prefs.LOG}{column_name}"] = [math.log(x) for x in result_df[column_name]]
    return result_df


def calculate_from_prev_row[T](calculation: Callable[[Row, Row], T]) -> Callable[[Row, Row], Optional[T]]:
    def wrapper(prev_row: Row, cur_row: Row):
        if cur_row.country == prev_row.country and cur_row.year - 1 == prev_row.year:
            return calculation(prev_row, cur_row)
        return None
    return wrapper


def calculate_gov_instability(df: pd.DataFrame, country: str, year: int) -> int:
    return len(df[(df[Cols.COUNTRY] == country) & (df[Cols.YEAR] < year) & (df[Cols.YEAR] >= year - GOV_INSTABILITY_LOOKBACK_YEARS) & (df[Cols.DUR] == 0)])
