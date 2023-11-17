from enum import StrEnum, Enum
import pandas as pd

# https://stackoverflow.com/questions/33690064/dynamically-create-an-enum-with-custom-values-in-python

# Merged dataset column names
class Column(StrEnum):
    COUNTRY = "country"
    YEAR = "year"
    GTYPE = "gov_type"
    POL = "polity"
    POL2 = "polity2"
    DUR = "durable"
    GOV_INSTABILITY = "gov_instability"
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
    RISK_TEST_2023 = "risk_test_2023"
    INVEST_RISK = "invest_risk"
    POL_RISK = "pol_risk"
    RISK = "risk"
    REG = "region"
    SUB_REG = "sub_region"
    COUNTRY_RISK = "country_risk"

    def get_description(self):
        return Description[f"{self.name}"]


# Prefixes
class Prefix(StrEnum):
    LOG = "log_"
    NORM = "norm_"
    NORM_LOG = "norm_log_"


class Description(StrEnum):
    GDP = "Constant dollar (2017) GDP per capita (billions)"
    GDP_PC = "Constant dollar (2017) GDP per capita (thousands)"
    LOG_GDP_PC = "Constant dollar (2017) GDP per capita (thousands, log(10))"
    INVEST = "Sum of investment data (billions)"
    DUR = "Years since regime change"
    NORM_RISK = "Risk factor (0..1, double)"
    POL2 = "Polity 2 score (-10..10, integer)"


class RiskClassification:
    def __init__(self, name: str, value: int, lower_bound: float, upper_bound: float):
        self.name = name
        self.value = value

        if lower_bound > upper_bound:
            raise Exception("lower_bound should be smaller or equal to upper_bound")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def condition(self, ser: pd.Series) -> pd.Series:
        return ((ser > self.lower_bound) & (ser <= self.upper_bound))


class RiskClassifications(object):
    # -0.1 to include a perfect score of 0

    # low = RiskClassification("low", 1, -0.1, 0.39)
    # medium = RiskClassification("medium", 2, 0.39, 0.69)
    # high = RiskClassification("high", 3, 0.69, 0.89)
    # critical = RiskClassification("critical", 4, 0.89, 1)

    # low = RiskClassification("low", 1, -0.1, 0.25)
    # medium = RiskClassification("medium", 2, 0.25, 0.5)
    # high = RiskClassification("high", 3, 0.5, 0.75)
    # critical = RiskClassification("critical", 4, 0.75, 1)

    low = RiskClassification("low", 1, -0.1, 0.333333)
    medium = RiskClassification("medium", 2, 0.333333, 0.666666)
    high = RiskClassification("high", 3, 0.666666, 1)

    @classmethod
    def __get_attributes(cls) -> dict:
        return {x: v for x, v in vars(cls).items() if type(v) is RiskClassification}

    @classmethod
    def get_names(cls) -> [str]:
        attrs = cls.__get_attributes()
        return [v.name for v in attrs.values()]

    @classmethod
    def get_values(cls) -> [str]:
        attrs = cls.__get_attributes()
        return [v.value for v in attrs.values()]

    @classmethod
    def get_conditions(cls) -> [callable]:
        attrs = cls.__get_attributes()
        return [v.condition for v in attrs.values()]
