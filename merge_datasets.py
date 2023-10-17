import numpy as np
import pandas as pd
from colorama import Fore, Style
from typing import Optional


def log_error(message: str):
    file = open("error.log", "a")
    file.write(message + "\n")
    file.close()


def main() -> None:
    print(Fore.GREEN + "Loading datasets..." + Style.RESET_ALL)
    polity_df = pd.read_excel("datasets/Polity5.xls")
    economic_df = pd.read_excel("datasets/IMFInvestmentandCapitalStockDataset2021.xlsx", sheet_name="Dataset")
    population_df = pd.read_excel("datasets/API_SP.POP.TOTL_DS2_en_excel_v2_5871620.xls", sheet_name="Data")

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

    print(Fore.GREEN + "Adding column: Sum of investment (sum_invest)..." + Style.RESET_ALL)
    df["sum_invest"] = df["igov_rppp"] + df["kgov_rppp"] + df["ipriv_rppp"] + df["kpriv_rppp"] + df["ippp_rppp"] + df["kppp_rppp"]

    print(Fore.GREEN + "Exporting to datasets/MergedDataset-v1.csv" + Style.RESET_ALL)
    df.to_csv("datasets/MergedDataset-v1.csv", index=False)
    print("(rows, columns):", df.shape)


if __name__ == '__main__':
    main()

