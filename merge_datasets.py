from colorama import Fore, Style
from functools import reduce
from modules import logger
from modules.helper_functions import *
from configuration import *
import os.path
from tqdm import tqdm


def main() -> None:
    print(Fore.GREEN + "Loading datasets..." + Style.RESET_ALL)
    with tqdm(total=4, ncols=100) as pbar:
        pbar.set_description("Loading 'Polity5' dataset")
        polity_df = pd.read_excel(os.path.join(DATASETS_PATH, "Polity5.xls"))
        pbar.update(1)

        pbar.set_description("Loading 'IMF' dataset")
        economic_df = pd.read_excel(os.path.join(DATASETS_PATH, "IMFInvestmentandCapitalStockDataset2021.xlsx"), sheet_name="Dataset")
        pbar.update(1)

        pbar.set_description("Loading 'Population' dataset")
        population_df = pd.read_excel(os.path.join(DATASETS_PATH, "API_SP.POP.TOTL_DS2_en_excel_v2_5871620.xls"), sheet_name="Data")
        pbar.update(1)

        pbar.set_description("Loading 'Continent' dataset")
        continent_df = pd.read_csv(os.path.join(DATASETS_PATH, "IMF_Countries_by_Continent.csv"), delimiter=";")
        pbar.update(1)
    pbar.close()

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
    polity_df = polity_df.replace(POLITY_COUNTRY_CONVERSIONS)
    population_df = population_df.replace(POPULATION_COUNTRY_CONVERSIONS)

    print(Fore.GREEN + "Removing rows with non-matching country names..." + Style.RESET_ALL)
    polity_countries = polity_df["country"].unique()
    economic_countries = economic_df["country"].unique()
    population_countries = population_df["Country Name"].unique()

    countries_not_in_polity = np.setdiff1d(economic_countries, polity_countries)
    countries_not_in_economic = np.setdiff1d(polity_countries, economic_countries)

    polity_df = polity_df[~polity_df["country"].isin(countries_not_in_polity)]
    economic_df = economic_df[~economic_df["country"].isin(countries_not_in_economic)]

    print(Fore.GREEN + "Adding column: Government instability (gov_instability)..." + Style.RESET_ALL)
    polity_df[Cols.GOV_INSTABILITY] = [
        calculate_gov_instability(polity_df, country, year)
        for country, year
        in tqdm(
            zip(polity_df[Cols.COUNTRY], polity_df[Cols.YEAR]),
            total=len(polity_df[Cols.COUNTRY]),
            ncols=100,
            desc="Processing")]

    print(Fore.GREEN + "Merging datasets..." + Style.RESET_ALL)
    df = pd.merge(polity_df, economic_df, how="inner", on=["country", "year"])
    df = pd.merge(df, continent_df, how="inner", on=["country"])
    df.columns = map(str.lower, df.columns)

    df_countries = economic_df["country"].unique()
    countries_not_in_population = np.setdiff1d(df_countries, population_countries)

    print(Fore.GREEN + "Dropping unused columns..." + Style.RESET_ALL)
    cols_to_drop = np.array(("cyear", "ccode", "scode", "flag", "xrreg", "xrcomp", "xropen", "xconst",
                             "parreg", "parcomp", "exrec", "exconst", "polcomp", "prior", "emonth", "eday", "eyear",
                             "eprec", "interim", "bmonth", "bday", "byear", "bprec", "post", "change", "d5", "sf",
                             "regtrans", "isocode", "ifscode", "igov_n", "kgov_n", "ipriv_n", "kpriv_n", "kppp_n",
                             "gdp_n"))
    df = df.drop(columns=cols_to_drop)
    print(Fore.YELLOW + "Columns dropped: ", cols_to_drop, Style.RESET_ALL)

    print(Fore.YELLOW + "Countries not in Polity5: ", countries_not_in_polity)
    print("Countries not in IMF: ", countries_not_in_economic)
    print("Countries not in Population: ", countries_not_in_population, Style.RESET_ALL)

    print(Fore.GREEN + "Adding column: Government Type (gov_typ)..." + Style.RESET_ALL)
    gov_conditions = [(df[Cols.POL2] > 5), (df[Cols.POL2] < -5), (df[Cols.POL2] >= -5) & (df[Cols.POL2] <= 5)]
    gov_options = ["democ", "autoc", "anoc"]
    df[Cols.GTYPE] = np.select(gov_conditions, gov_options)

    print(Fore.GREEN + "Adding column: Population (population)..." + Style.RESET_ALL)

    def get_population(country: str, year: int) -> Optional[int]:
        rows_df = population_df[population_df["Country Name"] == country]

        # rows_df should only contain 1 row
        if rows_df.shape[0] == 1:
            return population_df[population_df["Country Name"] == country].iloc[0][str(year)]

        logger.log_error(f"[get_population] Country: '{country}', Year: '{year}', found {rows_df.shape[0]} rows")
        return None

    df[Cols.POP] = [
        get_population(country, year)
        for country, year
        in tqdm(
            zip(df[Cols.COUNTRY], df[Cols.YEAR]),
            total=len(df[Cols.COUNTRY]),
            ncols=100,
            desc="Processing")]

    print(Fore.GREEN + "Adding column: Constant-Dollar GDP 2017 per Capita (GDP_rppp_pc)..." + Style.RESET_ALL)
    df[Cols.GDP_PC] = [(gdp_rpp * 1_000_000_000) / population for gdp_rpp, population in zip(df[Cols.GDP], df[Cols.POP])]

    set_to_default_columns = [Cols.IGOV, Cols.KGOV, Cols.IPRIV, Cols.KPRIV, Cols.IPPP, Cols.KPPP, Cols.FRAG]
    print(Fore.GREEN + f"Setting default values where NaN for columns {set_to_default_columns}..." + Style.RESET_ALL)
    df[set_to_default_columns] = df[set_to_default_columns].fillna(0)

    print(Fore.GREEN + "Adding column: Sum of investment (sum_invest)..." + Style.RESET_ALL)
    df[Cols.INVEST] = df[Cols.IGOV] + df[Cols.IPRIV] + df[Cols.IPPP]

    print(Fore.GREEN + "Adding column: Annual GDP per capita growth (gdp_rppp_pc_growth)..." + Style.RESET_ALL)
    shifted_df = df.shift(1)
    calculate_annual_gdp_growth = calculate_from_prev_row(lambda prev_row, cur_row: (cur_row.gdp_pc - prev_row.gdp_pc) / prev_row.gdp_pc * 100)
    df[Cols.GDP_PC_GR] = [
        calculate_annual_gdp_growth(
            Row.create(country=prev_country, year=prev_year, gdp_pc=prev_gdp_pc),
            Row.create(country=cur_country, year=cur_year, gdp_pc=cur_gdp_pc))
        for prev_country, prev_year, prev_gdp_pc, cur_country, cur_year, cur_gdp_pc
        in zip(
            shifted_df[Cols.COUNTRY], shifted_df[Cols.YEAR], shifted_df[Cols.GDP_PC],
            df[Cols.COUNTRY], df[Cols.YEAR], df[Cols.GDP_PC])]

    print(Fore.GREEN + "Selecting rows where (year > 1960)..." + Style.RESET_ALL)
    # Since "gdp_rppp_pc_growth" and "durable_changed" cannot be calculated from years before 1961,
    # given the data of our current datasets.
    df = df[df[Cols.YEAR] > 1960]

    drop_na_columns = [Cols.POL2, Cols.DUR, Cols.GDP_PC, Cols.GDP_PC_GR]
    print(Fore.YELLOW + f"Dropping NA values in columns {drop_na_columns}..." + Style.RESET_ALL)
    df = df.dropna(subset=drop_na_columns)
    df = df[df[Cols.INVEST] != 0]

    log_columns = [Cols.GDP_PC, Cols.GDP, Cols.INVEST]
    print(Fore.GREEN + f"Adding Math.Log columns for columns {log_columns}..." + Style.RESET_ALL)
    df = reduce(log_column, log_columns, df)

    columns_to_normalize = [Cols.DUR, Cols.GDP_PC, Cols.GDP_PC_GR, Cols.GDP, Cols.POL2, Cols.INVEST, Cols.GOV_INSTABILITY]
    columns_to_normalize += map(lambda s: f"log_{s}", log_columns)
    print(Fore.GREEN + f"Adding normalized columns (min: {A}, max: {B}) for columns {columns_to_normalize}..." + Style.RESET_ALL)
    df = reduce(normalize_column, columns_to_normalize, df)

    print(Fore.GREEN + f"Adding risk columns: {Cols.INVEST_RISK}, {Cols.POL_RISK}, {Cols.RISK} and {Prefs.NORM + Cols.RISK}..." + Style.RESET_ALL)
    df[Cols.INVEST_RISK] = -(df[Prefs.NORM_LOG + Cols.GDP_PC] + df[Prefs.NORM_LOG + Cols.GDP] + df[Prefs.NORM_LOG + Cols.INVEST])
    df[Cols.POL_RISK] = -((abs(df[Cols.POL2]) / 10) + df[Prefs.NORM + Cols.DUR] - (df[Cols.FRAG] / 3) - (df[Prefs.NORM + Cols.GOV_INSTABILITY]))
    df[Cols.RISK] = df[Cols.INVEST_RISK] + df[Cols.POL_RISK]
    df = normalize_column(df, Cols.RISK)

    # TODO: Estimate empty values

    print(Fore.GREEN + f"Exporting to {MERGED_DATASET_PATH}" + Style.RESET_ALL)
    with tqdm(total=1, ncols=100) as pbar:
        pbar.set_description("Exporting")
        df.to_excel(
            MERGED_DATASET_PATH,
            index=False,
            sheet_name="Data")
        pbar.update(1)
        pbar.close()
    print("(rows, columns):", df.shape)


if __name__ == '__main__':
    main()
