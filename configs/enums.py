from enum import StrEnum, Enum
import pandas as pd
import itertools

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
    GOV_INSTABILITY = "Government Instability (TBA)"
    IGOV = "General government investment in billions constant dollars (2017)"
    IPRIV = "Private investment in billions constant dollars (2017)"


class RiskClassification:
    def __init__(self, name: str, value: int, lower_bound: float, upper_bound: float):
        self.name = name
        self.value = value

        if lower_bound > upper_bound:
            raise Exception(f"lower_bound [{lower_bound}] should be smaller or equal to upper_bound [{upper_bound}]")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def condition(self, ser: pd.Series) -> pd.Series:
        return ((ser > self.lower_bound) & (ser <= self.upper_bound))


class RiskClassifications:
    def __init__(self, classifications: [RiskClassification]):
        for c in classifications:
            self.__dict__[c.name] = c

    def get_attributes(self) -> dict:
        return {x: v for x, v in vars(self).items() if type(v) is RiskClassification}

    def get_names(self) -> [str]:
        attrs = self.get_attributes()
        return [v.name for v in attrs.values()]

    def get_values(self) -> [str]:
        attrs = self.get_attributes()
        return [v.value for v in attrs.values()]

    def get_conditions(self) -> [callable]:
        attrs = self.get_attributes()
        return [v.condition for v in attrs.values()]


__amount_of_classes = 9
__class_names = ["low", "medium", "high"]

__step = 1 / __amount_of_classes
__names_cart_product = itertools.product(__class_names, range(0, __amount_of_classes // len(__class_names)))
__names = [f"{l}_{r}" for l, r in __names_cart_product]
__classifications = [
    RiskClassification(name=name, value=value, lower_bound=value * __step, upper_bound=(value+1) * __step)
    for name, value
    in zip(__names, range(0, __amount_of_classes))]

RISKCLASSIFICATIONS = RiskClassifications(__classifications)


def get_amount_of_classes():
    return __amount_of_classes
