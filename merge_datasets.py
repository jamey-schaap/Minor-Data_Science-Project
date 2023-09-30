import numpy as np
import pandas as pd


def main() -> None:
    print("Loading datasets...")
    polity_df = pd.read_excel("datasets/Polity5.xls")
    economic_df = pd.read_excel("datasets/IMFInvestmentandCapitalStockDataset2021.xlsx", sheet_name="Dataset")

    polity_df = polity_df[polity_df["year"] >= 1960]

    REMOVABLE_COUNTRIES = [
        "Yugoslavia", # 1918 - 1992
        "Yemen", "Yemen North", "Yemen South", # Used to be split, now is together. Missing IMF data.
        "Cuba", # Seems like it's missing from IMF.
        "Czechoslovakia", # Difficult to match it with 100% guarantee to IMF.
        "Serbia and Montenegro", # Missing IMF data. Just 4 rows of polity data.
        "Kosovo", # Missing IMF data. Just 11 rows of polity data.
    ]

    REPLACEABLE_COUNTRIES = {
        "Cabo Verde": "Cape Verde",
        "Bosnia": "Bosnia and Herzegovina",
        # https://theculturetrip.com/africa/congo/articles/congo-vs-d-r-c-whats-the-difference
        "Congo-Brazzaville": "Congo, Republic of",
        "Congo Brazzaville": "Congo, Republic of",
        "Congo Kinshasa": "Congo, Democratic Republic of the",
        "Cote D'Ivoire": "Côte d'Ivoire",
        "Ivory Coast": "Côte d'Ivoire",
        "Myanmar (Burma)": "Myanmar",
        "Gambia, The": "Gambia",
        "Timor-Leste": "Timor Leste",
        "Lao P.D.R.": "Laos",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Swaziland": "Eswatini",
        "North Macedonia": "Macedonia",
        "Montenegro, Rep. of": "Montenegro",
        "UAE": "United Arab Emirates",
    }

    polity_df = polity_df[~polity_df["country"].isin(REMOVABLE_COUNTRIES)]

    polity_df = polity_df.replace(REPLACEABLE_COUNTRIES)
    economic_df = economic_df.replace(REPLACEABLE_COUNTRIES)

    print("Checking the difference country names...")
    polity_countries = polity_df["country"].unique()
    economic_countries = economic_df["country"].unique()
    countries_not_in_polity = np.setdiff1d(economic_countries, polity_countries)
    countries_not_in_economic = np.setdiff1d(polity_countries, economic_countries)

    print("Removing rows with non-matching country names...")
    polity_df = polity_df[~polity_df["country"].isin(countries_not_in_polity)]
    economic_df = economic_df[~economic_df["country"].isin(countries_not_in_economic)]

    print("Merging datasets...")
    df = pd.merge(polity_df, economic_df, how="right", on=["country", "year"])

    print("Adding government type (gov_typ) column...")
    gov_conditions = [(df["polity2"] > 5), (df["polity2"] < -5), (df["polity2"] >= -5) & (df["polity2"] <= 5)]
    gov_options = ["democ", "autoc", "anoc"]
    df["gov_type"] = np.select(gov_conditions, gov_options)

    print("Exporting to datasets/MergedDataset-v1.csv")
    df.to_csv("datasets/MergedDataset-v1.csv", index=False)
    print("(rows, columns):", df.shape)


if __name__ == '__main__':
    main()

