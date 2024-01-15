import os.path
from configs.enums import get_amount_of_classes

# Paths
_project_dir = ".."
_datasets_path = os.path.join(_project_dir, "datasets")
_out_path = os.path.join(_project_dir, "out")

# The amount of classes, thus version can be configured at:
# src\configs\data.py; line 96; __amount_of_classes
VERSION = f"RawData.{get_amount_of_classes()}c"

DATASETS_PATH = os.path.join(os.getcwd(), _datasets_path)
OUT_PATH = os.path.join(os.getcwd(), _out_path)
MODELS_PATH = os.path.join(os.getcwd(), _out_path, "models")
MERGED_DATASET_PATH = os.path.join(os.getcwd(), _datasets_path, f"MergedDataset-V.{VERSION}.xlsx")
MACHINE_LEARNING_DATASET_PATH = os.path.join(os.getcwd(), _datasets_path, f"MachineLearning-Dataset-V.{VERSION}.xlsx")

# Calculation gov_instability
GOV_INSTABILITY_LOOKBACK_YEARS = 60

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
