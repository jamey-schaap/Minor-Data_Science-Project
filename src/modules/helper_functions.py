from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from src.configs.data import A, B, GOV_INSTABILITY_LOOKBACK_YEARS
from src.configs.enums import Column, Prefix


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
            durable: int | None = None) -> Row:
        """
        Creates a Row object.
        :param country: str, A country's name for a given row.
        :param year: str, The year of a given row.
        :param gdp_pc: The GDP per capita of a given row.
        :param durable: The durability score of a given row.
        :return: Row, A (partial) Row Object.
        """
        return Row(country, year, gdp_pc, durable)


def normalize(x: int | float, x_min: int | float, x_max: int | float) -> float:
    """
    Normalizes a value.
    :param x: int | float, The value to normalize.
    :param x_min: int | float, The minimum x value of a given series.
    :param x_max: int | float, The maximum x value of a given series.
    :return: float, The normalized value.
    """
    return A + (((x - x_min) * (B - A)) / (x_max - x_min))


def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalizes a column | pandas.Series.
    :param df: pandas.Dataframe, A dataframe containing the column to normalize.
    :param column_name: str, The name of the column to normalize.
    :return: pandas.Dataframe, A dataframe where the given column is normalized.
    """
    result_df = df.copy()

    column_min = np.nanmin(result_df[column_name])
    column_max = np.nanmax(result_df[column_name])
    result_df[f"{Prefix.NORM}{column_name}"] = [normalize(x, column_min, column_max) for x in result_df[column_name]]

    return result_df


def log10_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Computes log10 over a given column | pandas.Series.
    :param df: pandas.Dataframe, A dataframe containing the column to normalize.
    :param column_name: str, The name of the column to normalize.
    :return: pandas.Dataframe, A dataframe where log10 is computed over the given column.
    """
    result_df = df
    result_df[f"{Prefix.LOG}{column_name}"] = [
        x if x == 0
        else math.log10(x)
        for x in result_df[column_name]]
    return result_df


# def calculate_from_prev_row(calculation: Callable[[Row, Row], T]) -> Callable[[Row, Row], Optional[T]]:
def calculate_from_prev_row(calculation):
    """
    A HOF which performs a calculation based on the current and previous row. Generic T.
    :param calculation: Callable[[Row, Row], T], The calculation that will be performed,
    :return: Callable[[Row, Row], Optional[T]], The HOF.
    """
    def wrapper(prev_row: Row, cur_row: Row):
        """
        Performs a calculation based on the current and previous row.
        :param prev_row: Row, The previous row.
        :param cur_row: Row, The current row.
        :return: T | None, The results of the calculation or None.
        """
        if cur_row.country == prev_row.country and cur_row.year - 1 == prev_row.year:
            return calculation(prev_row, cur_row)
        return None
    return wrapper


def calculate_gov_instability(df: pd.DataFrame, country: str, year: int) -> int:
    """
    Calculates the government instability for a given row.
    :param df: pandas.Dataframe, Containing all the rows.
    :param country: str, A country's name.
    :param year: int, The year of the row where the calculation will be performed on.
    :return: int, The instability score of a given country of a given year.
    """
    return len(df[(df[Column.COUNTRY] == country) & (df[Column.YEAR] < year) &
                  (df[Column.YEAR] >= year - GOV_INSTABILITY_LOOKBACK_YEARS) & (df[Column.DUR] == 0)])
