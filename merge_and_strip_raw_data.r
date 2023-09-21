library("readxl")

polity_data <- read_excel("datasets/Polity5.xls")
polity_df <- data.frame(polity_data)

economic_data <- read_excel("datasets/IMFInvestmentandCapitalStockDataset2021.xlsx", sheet="Dataset")
economic_df <- data.frame(economic_data)

print("Removing useless columns")

# Only keep relevant columns.
polity_columns <- c("scode", "country", "year", "polity", "polity2", "durable")
polity_df <- polity_df[polity_columns]

print("Merging data")

df <- merge(polity_df, economic_df, by.x=c("year", "scode"), by.y=c("year", "isocode"))
colnames(df)[colnames(df) == "country.x"] <- "country"
df$country.y <- NULL

write.csv(df, "datasets/1-MergedDataset.csv")

print("Done!")
