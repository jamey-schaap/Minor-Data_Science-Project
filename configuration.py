from enum import StrEnum

# Paths
DATASETS_PATH = "datasets"
MERGED_DATASET_PATH = f"{DATASETS_PATH}/MergedDataset-v1.csv"


# Merged dataset column names
class Cols(StrEnum):
    COUNTRY = "country"
    YEAR = "year"
    GTYPE = "gov_type"
    POL = "polity"
    POL2 = "polity2"
    DUR = "durable"
    DUR_CH = "durable_changed"
    GDP = "gdp_rppp"
    GDP_PC = "gdp_rppp_pc"
    GDP_PC_GR = "gdp_rppp_pc_growth"
    INVEST = "sum_invest"
    POP = "population"
    IGOV = "igov_rppp"
    KGOV = "kgov_rppp"
    IPRIV = "ipriv_rppp"
    KPRIV = "kpriv_rppp"
    IPPP = "ippp_rppp"
    KPPP = "kppp_rppp"
    FRAG = "fragment"


# Normalization
A = 0
B = 10


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