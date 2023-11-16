from enum import StrEnum

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