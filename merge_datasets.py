import math

import numpy as np
import pandas as pd
from colorama import Fore, Style
from typing import Optional, Callable
from dataclasses import dataclass
from functools import reduce

A = 0
B = 10


def normalize(x: int | float, x_min: int | float, x_max: int | float) -> float:
    return A + (((x - x_min) * (B - A)) / (x_max - x_min))


def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = df

    column_min = np.nanmin(result_df[column_name])
    column_max = np.nanmax(result_df[column_name])
    result_df[f"norm_{column_name}"] = [normalize(x, column_min, column_max) for x in result_df[column_name]]

    return result_df


def log_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    result_df = df
    result_df[f"log_{column_name}"] = [math.log(x) for x in result_df[column_name]]
    return result_df


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


def calculate_from_prev_row[T](calculation: Callable[[Row, Row], T]) -> Callable[[Row, Row], Optional[T]]:
    def wrapper(prev_row: Row, cur_row: Row):
        if cur_row.country == prev_row.country and cur_row.year - 1 == prev_row.year:
            return calculation(prev_row, cur_row)
        return None
    return wrapper


def log_error(message: str):
    with open("error.log", "a") as file:
        file.write(message + "\n")


def main() -> None:
    print(Fore.GREEN + "Loading datasets..." + Style.RESET_ALL)
    polity_df = pd.read_excel("datasets/Polity5.xls")
    economic_df = pd.read_excel("datasets/IMFInvestmentandCapitalStockDataset2021.xlsx", sheet_name="Dataset")
    population_df = pd.read_excel("datasets/API_SP.POP.TOTL_DS2_en_excel_v2_5871620.xls", sheet_name="Data")

    print(Fore.GREEN + "Selecting rows where (1960 <= year <= 2018)..." + Style.RESET_ALL)
    polity_df = polity_df[
        (polity_df["year"] >= 1960)
        & (polity_df["year"] <= 2018)]
    economic_df = economic_df[
        (economic_df["year"] >= 1960)
        & (economic_df["year"] <= 2018)]

    print(Fore.GREEN + "Removing conflicting rows..." + Style.RESET_ALL)
    # USSR started in December 1922
    polity_df = polity_df.drop(polity_df[
                                   (polity_df["country"] == "USSR")
                                   & (polity_df["year"] == 1922)].index)
    # YGS Yugoslavia seems to be correct (versus YUG)
    polity_df = polity_df.drop(polity_df[
                                   (polity_df["country"] == "Yugoslavia")
                                   & (polity_df["year"] == 1991)
                                   & (polity_df["scode"] == "YUG")].index)

    print(Fore.GREEN + "Replacing country names..." + Style.RESET_ALL)
    polity_country_conversions = {
        "Myanmar (Burma)": "Myanmar",
        "Gambia": "Gambia, The",
        "Cape Verde": "Cabo Verde",
        "Bosnia": "Bosnia and Herzegovina",
        "Timor Leste": "Timor-Leste",
        "UAE": "United Arab Emirates",
        "Gran Colombia": "Colombia",
        "Cote D'Ivoire": "Côte d'Ivoire",
        "Ivory Coast": "Côte d'Ivoire",
        "USSR": "Russia",
        "Congo Brazzaville": "Congo, Republic of",
        "Congo-Brazzaville": "Congo, Republic of",
        "Congo Kinshasa": "Congo, Democratic Republic of the",
        "Kyrgyzstan": "Kyrgyz Republic",
        "Laos": "Lao P.D.R.",
        "Swaziland": "Eswatini",
        "United Province CA": "Canada",
        "Serbia and Montenegro": "Serbia"
    }
    population_country_conversions = {
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Russian Federation": "Russia",
        "Congo, Rep.": "Congo, Republic of",
        "Congo, Dem. Rep.": "Congo, Democratic Republic of the",
        "Lao PDR": "Lao P.D.R.",
        "Czechia": "Czech Republic",
        "Egypt, Arab Rep.": "Egypt",
        "Hong Kong SAR, China": "Hong Kong SAR",
        "Iran, Islamic Rep.": "Iran",
        "Macao SAR, China": "Macao SAR",
        "Micronesia, Fed. Sts.": "Micronesia",
        "Syrian Arab Republic": "Syria",
        "Turkiye": "Turkey",
        "Venezuela, RB": "Venezuela",
        "Yemen, Rep.": "Yemen"
    }

    polity_df = polity_df.replace(polity_country_conversions)
    population_df = population_df.replace(population_country_conversions)

    print(Fore.GREEN + "Removing rows with non-matching country names..." + Style.RESET_ALL)
    polity_countries = polity_df["country"].unique()
    economic_countries = economic_df["country"].unique()
    population_countries = population_df["Country Name"].unique()

    countries_not_in_polity = np.setdiff1d(economic_countries, polity_countries)
    countries_not_in_economic = np.setdiff1d(polity_countries, economic_countries)

    polity_df = polity_df[~polity_df["country"].isin(countries_not_in_polity)]
    economic_df = economic_df[~economic_df["country"].isin(countries_not_in_economic)]

    print(Fore.GREEN + "Merging datasets..." + Style.RESET_ALL)
    df = pd.merge(polity_df, economic_df, how="inner", on=["country", "year"])

    df_countries = economic_df["country"].unique()
    countries_not_in_population = np.setdiff1d(df_countries, population_countries)

    print(Fore.GREEN + "Dropping unused columns..." + Style.RESET_ALL)
    cols_to_drop = np.array(("cyear", "ccode", "scode", "flag", "fragment", "xrreg", "xrcomp", "xropen", "xconst",
                             "parreg", "parcomp", "exrec", "exconst", "polcomp", "prior", "emonth", "eday", "eyear",
                             "eprec", "interim", "bmonth", "bday", "byear", "bprec", "post", "change", "d5", "sf",
                             "regtrans", "isocode", "ifscode", "igov_n", "kgov_n", "ipriv_n", "kpriv_n", "kppp_n",
                             "GDP_n"))
    df = df.drop(columns=cols_to_drop)
    print(Fore.YELLOW + "Columns dropped: ", cols_to_drop, Style.RESET_ALL)

    print(Fore.YELLOW + "Countries not in Polity5: ", countries_not_in_polity)
    print("Countries not in IMF: ", countries_not_in_economic)
    print("Countries not in Population: ", countries_not_in_population, Style.RESET_ALL)

    print(Fore.GREEN + "Adding column: Government Type (gov_typ)..." + Style.RESET_ALL)
    gov_conditions = [(df["polity2"] > 5), (df["polity2"] < -5), (df["polity2"] >= -5) & (df["polity2"] <= 5)]
    gov_options = ["democ", "autoc", "anoc"]
    df["gov_type"] = np.select(gov_conditions, gov_options)

    print(Fore.GREEN + "Adding column: Population (population)..." + Style.RESET_ALL)

    def get_population(country: str, year: int) -> Optional[int]:
        rows_df = population_df[population_df["Country Name"] == country]

        # rows_df should only contain 1 row
        if rows_df.shape[0] == 1:
            return population_df[population_df["Country Name"] == country].iloc[0][str(year)]

        log_error(f"[get_population] Country: '{country}', Year: '{year}', found {rows_df.shape[0]} rows")
        return None

    df["population"] = [get_population(country, year) for country, year in zip(df["country"], df["year"])]

    print(Fore.GREEN + "Adding column: Constant-Dollar GDP 2017 per Capita (GDP_rppp_pc)..." + Style.RESET_ALL)
    df["GDP_rppp_pc"] = [(gdp_rpp * 1_000_000_000) / population for gdp_rpp, population in zip(df["GDP_rppp"], df["population"])]

    set_to_default_columns = ["igov_rppp", "kgov_rppp", "ipriv_rppp", "kpriv_rppp", "ippp_rppp", "kppp_rppp"]
    print(Fore.GREEN + f"Setting default values where NaN for columns {set_to_default_columns}..." + Style.RESET_ALL)
    df[set_to_default_columns] = df[set_to_default_columns].fillna(0)

    print(Fore.GREEN + "Adding column: Sum of investment (sum_invest)..." + Style.RESET_ALL)
    df["sum_invest"] = df["igov_rppp"] + df["kgov_rppp"] + df["ipriv_rppp"] + df["kpriv_rppp"] + df["ippp_rppp"] + df["kppp_rppp"]

    print(Fore.GREEN + "Adding column: Durable changed (durable_changed)..." + Style.RESET_ALL)
    shifted_df = df.shift(1)
    calculate_durable_changed = calculate_from_prev_row(lambda prev_row, cur_row: cur_row.durable - prev_row.durable == 0)
    # True when durable has changed back to 0.
    df["durable_changed"] = [
        calculate_durable_changed(
            Row.create(country=prev_country, year=prev_year, durable=prev_durable),
            Row.create(country=cur_country, year=cur_year, durable=cur_durable))
        for prev_country, prev_year, prev_durable, cur_country, cur_year, cur_durable
        in zip(
            shifted_df["country"], shifted_df["year"], shifted_df["durable"],
            df["country"], df["year"], df["durable"])]

    print(Fore.GREEN + "Adding column: Annual GDP per capita growth (gdp_rppp_pc_growth)..." + Style.RESET_ALL)
    calculate_annual_gdp_growth = calculate_from_prev_row(lambda prev_row, cur_row: (cur_row.gdp_pc - prev_row.gdp_pc) / prev_row.gdp_pc * 100)
    df["gdp_rppp_pc_growth"] = [
        calculate_annual_gdp_growth(
            Row.create(country=prev_country, year=prev_year, gdp_pc=prev_gdp_pc),
            Row.create(country=cur_country, year=cur_year, gdp_pc=cur_gdp_pc))
        for prev_country, prev_year, prev_gdp_pc, cur_country, cur_year, cur_gdp_pc
        in zip(
            shifted_df["country"], shifted_df["year"], shifted_df["GDP_rppp_pc"],
            df["country"], df["year"], df["GDP_rppp_pc"])]

    print(Fore.GREEN + "Selecting rows where (year > 1960)..." + Style.RESET_ALL)
    # Since "gdp_rppp_pc_growth" and "durable_changed" cannot be calculated from years before 1961,
    # given the data of our current datasets.
    df = df[df["year"] > 1960]

    drop_na_columns = ["polity2", "durable", "GDP_rppp_pc", "gdp_rppp_pc_growth", "durable_changed"]
    print(Fore.GREEN + f"Dropping NA values in columns {drop_na_columns}..." + Style.RESET_ALL)
    df = df.dropna(subset=drop_na_columns)
    df = df[df["sum_invest"] != 0]

    log_columns = ["GDP_rppp_pc", "sum_invest"]
    print(Fore.GREEN + f"Adding Math.Log columns for columns {log_columns}..." + Style.RESET_ALL)
    df = reduce(log_column, log_columns, df)

    columns_to_normalize = ["durable", "GDP_rppp_pc", "gdp_rppp_pc_growth", "polity2", "sum_invest"]
    columns_to_normalize += map(lambda s: f"log_{s}", log_columns)
    print(Fore.GREEN + f"Adding normalized columns (min: {A}, max: {B}) for columns {columns_to_normalize}..." + Style.RESET_ALL)
    df = reduce(normalize_column, columns_to_normalize, df)

    print(Fore.GREEN + "Exporting to datasets/MergedDataset-v1.csv" + Style.RESET_ALL)
    df.to_csv("datasets/MergedDataset-v1.csv", index=False)
    print("(rows, columns):", df.shape)


if __name__ == '__main__':
    main()

