from __future__ import annotations
from enum import StrEnum, Enum
import pandas as pd
import itertools


class Column(StrEnum):
    """
    Standardized column names for following datasets:
        MergedDataset-V.*.*c.xlsx
        MachineLearning-V.*.*c.

    # Examples:
        1. dataframe[Column.COUNTRY_RISK] => Series
    """
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

    def get_description(self) -> Description:
        """
        Gets the Description associated with the used Column instance.
        :return: A Description enum.
        """
        return Description[f"{self.name}"]


class Prefix(StrEnum):
    """
    Standardized prefixes for the Column Enum, for the following datasets:
           MergedDataset-V.*.*c.xlsx
           MachineLearning-V.*.*c.xlsx
    # Examples:
        1. Prefix.NORM_LOG + Column.COUNTRY_RISK = norm_log_country_risk
        2. dataframe[Prefix.NORM_LOG + Column.COUNTRY_RISK] => Series
    """
    LOG = "log_"
    NORM = "norm_"
    NORM_LOG = "norm_log_"


class Description(StrEnum):
    """Standardized descriptions associated with the Column enum."""
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
    """The risk classification format."""
    def __init__(self, name: str, value: int, lower_bound: float, upper_bound: float) -> None:
        """
        :param name: str, The name of the risk classification.
        :param value: int, The classification (machine learning) value.
        :param lower_bound: float, The lower boundary/fence of the risk classification.
        :param upper_bound: float, The upper boundary/fence of the risk classification.
        """
        self.name = name
        self.value = value

        if lower_bound > upper_bound:
            raise Exception(f"lower_bound [{lower_bound}] should be smaller or equal to upper_bound [{upper_bound}]")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def condition(self, ser: pd.Series) -> pd.Series:
        """
        Checks for each row if they fall within the boundaries of the given instance.
        :param ser: pandas.Series, A Pandas Series.
        :return: pandas.Series, A boolean map stating for each row if they fall within the boundaries of the given
        instance.
        """
        return ((ser > self.lower_bound) & (ser <= self.upper_bound))


class RiskClassifications:
    """A datastructure which holds multiple risk classifications (type: RiskClassification)"""
    def __init__(self, classifications: [RiskClassification]) -> None:
        """
        :param classifications: [RiskClassification], A list of RiskClassifications which the datastructure will
        contain.
        """
        classifications.sort(key=lambda c: c.value)

        # Since the lower boundary is excluded from the condition, 0 will not be included in the boundary.
        # To omit this, a small value is subtracted from 0.
        if len(classifications) > 0 and classifications[0].lower_bound == 0:
            classifications[0].lower_bound -= 0.000001

        for c in classifications:
            self.__dict__[c.name] = c

    def get_attributes(self) -> dict[str, RiskClassification]:
        """
        Gets all attributes i.e. RiskClassifications contained in the instance.
        :return: dict[str, RiskClassification], A dictionary containing each attribute i.e. RiskClassification
        """
        return {x: v for x, v in vars(self).items() if type(v) is RiskClassification}

    def get_names(self) -> [str]:
        """
        Gets the names of all attributes i.e. RiskClassifications
        :return: [str], A list of all the attributes i.e. RiskClassifications names.
        """
        attrs = self.get_attributes()
        return [v.name for v in attrs.values()]

    def get_values(self) -> [int]:
        """
        Gets the values of all attributes i.e. RiskClassifications
        :return: [int], A list of all the attributes i.e. RiskClassifications values.
        """
        attrs = self.get_attributes()
        return [v.value for v in attrs.values()]

    def get_conditions(self) -> [callable]:
        """
        Gets the conditions of all attributes i.e. RiskClassifications
        :return: [callable], A list of all the attributes i.e. RiskClassifications conditions.
        """
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

# A constant datastructure containing all RiskClassifications with respect to the configuration above
RISKCLASSIFICATIONS = RiskClassifications(__classifications)


def get_amount_of_classes():
    """
    Gets the amount of classes/labels.
    :return: int, The amount of classes/labels.
    """
    return __amount_of_classes
