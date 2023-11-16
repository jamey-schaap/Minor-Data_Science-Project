import os.path

# Paths
DATASETS_PATH = "datasets"
MERGED_DATASET_PATH = os.path.join(os.getcwd(), DATASETS_PATH, "MergedDataset-V2.xlsx")

# Calculation gov_instability
GOV_INSTABILITY_LOOKBACK_YEARS = 200

# Normalization
A = 0
B = 1


# Country conversion
POLITY_COUNTRY_CONVERSIONS = {
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

POPULATION_COUNTRY_CONVERSIONS = {
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