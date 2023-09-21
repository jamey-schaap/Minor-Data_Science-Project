library("readxl")

polity_data <- read_excel("datasets/Polity5.xls")
polity_df <- data.frame(polity_data)

economic_data <- read_excel("datasets/IMFInvestmentandCapitalStockDataset2021.xlsx", sheet="Dataset")
economic_df <- data.frame(economic_data)

country_codes_data <- read.csv("datasets/ISO-3166-Country-Codes.csv")
country_codes_df <- data.frame(country_codes_data)

print("Removing unused countries from polity_df")
# The Polity5 dataset contains countries that IMF dataset does contain
diff_in_countries <- setdiff(unique(polity_df$country), unique(economic_df$country))
polity_df <- polity_df[!polity_df$country %in% diff_in_countries,]

print("Replacing country codes")

for (index in seq_len(nrow(polity_df))) {
  current_row <- polity_df[index,]
  if (!current_row$country %in% diff_in_countries) {
    country_data <- country_codes_df[grepl(current_row$country, country_codes_df$name, fixed = TRUE),]
    # Some countries might have 2 codes, hence why index 1 is specified
    polity_df[index,]$scode <- country_data$alpha.3[1]
  }
}

for (index in seq_len(nrow(economic_df))) {
  current_row <- economic_df[index,]
  if (!current_row$country %in% diff_in_countries) {
    country_data <- country_codes_df[grepl(current_row$country, country_codes_df$name, fixed = TRUE),]
    # Some countries might have 2 codes, hence why index 1 is specified
    economic_df[index,]$isocode <- country_data$alpha.3[1]
  }
}

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
