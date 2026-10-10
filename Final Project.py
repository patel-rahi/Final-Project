## Final Project - Python Programming II

## Research Question: 
## To what extent is a country's digital economy associated with the strictness of its data privacy laws?

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# populate datasets
comparitech_df = pd.read_excel("datasets/CompariTech Dataset.xlsx")
owid_df = pd.read_csv("datasets/Our World in Data Dataset.csv")
unctad_df = pd.read_csv("datasets/UNCTAD Dataset.csv")
iso_df = pd.read_csv("datasets/ISO Codes.csv")
population_df = pd.read_csv("datasets/WBG Population Dataset.csv", skiprows=4)

# ----------------------------------------------------------------------------------------------------------
# Data Cleaning of Comparitech Dataset
# ----------------------------------------------------------------------------------------------------------
comparitech_df.rename(columns={"Unnamed: 0": "country"}, inplace=True)
comparitech_df["country"] = comparitech_df["country"].str.strip()
comparitech_df = comparitech_df.drop(comparitech_df[comparitech_df["country"] == "European Union"].index)
comparitech_df = comparitech_df.drop(comparitech_df[comparitech_df["country"] == "International"].index)
comparitech_df = comparitech_df.reset_index(drop=True)

comparitech_df["country"] = comparitech_df["country"].replace({
    "UK": "United Kingdom of Great Britain and Northern Ireland",
    "Netherlands": "Netherlands, Kingdom of the",
    "Czech Republic": "Czechia",
    "Taiwan": "Taiwan, Province of China",
    "US": "United States of America",
    "Russia": "Russian Federation",
})

comparitech_df = comparitech_df.merge(
    iso_df[["name", "alpha-3", "country-code"]],
    left_on="country",
    right_on="name",
    how="left"
).drop(columns="name").rename(columns={
    "alpha-3": "iso_code",
    "country-code": "numeric_code"
})

unctad_df["code"] = (
    unctad_df["code"]
    .astype("Int64")
    .astype("string")
    .str.zfill(3)
)

comparitech_df["numeric_code"] = comparitech_df["numeric_code"].astype("string")


# ----------------------------------------------------------------------------------------------------------
# Merge Datasets
# ----------------------------------------------------------------------------------------------------------
comparitech_df = comparitech_df.merge(
    unctad_df,
    left_on="numeric_code",
    right_on="code",
    how="left"
).drop(columns="code")

owid_df = owid_df[owid_df["Year"] == 2024]
comparitech_df = comparitech_df.merge(
    owid_df,
    left_on="iso_code",
    right_on="Code",
    how="left"
).drop(columns="Code")

population_df = population_df[["Country Code", "2024"]]
comparitech_df = comparitech_df.merge(
    population_df,
    left_on="iso_code",
    right_on="Country Code",
    how="left"
).drop(columns="Country Code")

comparitech_df.to_csv("final.csv", index=False)


# ----------------------------------------------------------------------------------------------------------
# Data Cleaning of Final Merged Dataset
# ----------------------------------------------------------------------------------------------------------
final_df = pd.read_csv("final.csv", dtype={"numeric_code": str})

final_df.columns = (
    final_df.columns
    .str.strip()
    .str.lower()
    .str.replace(",", "", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

final_df = final_df.rename(columns={
    "country_x": "country",
    "number_of_people_using_the_internet": "internet_users",
    "surveillance_of_medical_financial_and_movement": "surveillance_medical_financial_movement",
})

final_df = final_df.drop(columns=["country_y", "entity", "year"])

final_df = final_df.drop_duplicates().drop_duplicates(subset="iso_code")

final_df["numeric_code"] = final_df["numeric_code"].str.zfill(3)          # 40 -> "040"
final_df["internet_users"] = final_df["internet_users"].round().astype("Int64")

final_df = final_df.dropna().reset_index(drop=True)

final_df["country"] = final_df["country"].replace({
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Netherlands, Kingdom of the": "Netherlands",
    "Russian Federation": "Russia",
})


# ----------------------------------------------------------------------------------------------------------
# Data Manipulation
# ----------------------------------------------------------------------------------------------------------
# Percentage of Internet Users per Country
final_df["population"] = pd.to_numeric(final_df["2024"])
final_df["internet_pct"] = (final_df["internet_users"] / final_df["population"] * 100).round(2)

# Region Grouping
region_cols = {"africa": "Africa", 
               "asia_and_oceania": "Asia & Oceania",
               "europe": "Europe", 
               "northern_america": "Northern America",
               "latin_america_and_caribbean": "Latin America & Caribbean"}
final_df["region"] = final_df[list(region_cols)].idxmax(axis=1).map(region_cols)

# Number of Laws Grouping
law_cols = ["electronic_transactions", "consumer_protection", "privacy_and_data_protection", "cybercrime", "indirect_taxation"]
final_df["laws_count"] = (final_df[law_cols] == "Legislation").sum(axis=1)
final_df["law_group"] = np.where(final_df["laws_count"] == 5, "All 5 laws", "Missing 1+ laws")

# Top 15 Countries by Percentage of Internet Users
top15_internet = (final_df.nlargest(15, "internet_pct")[["country", "region", "internet_pct"]]
                  .round(2).reset_index(drop=True))
top15_internet.index = top15_internet.index + 1
top15_internet.index.name = "rank"
print(top15_internet)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Standardize Scores
# ----------------------------------------------------------------------------------------------------------
def to_100(series):
    return (series - series.min()) / (series.max() - series.min()) * 100

# privacy_score = standardize the total score from Comparitech (0-100)
final_df["privacy_score"] = to_100(final_df["total"])

# law_score = share of the 5 UNCTAD laws a country has (0-100)
law_cols = ["electronic_transactions", "consumer_protection",
            "privacy_and_data_protection", "cybercrime", "indirect_taxation"]
final_df["law_score"] = (final_df[law_cols] == "Legislation").sum(axis=1) / len(law_cols) * 100

# users_score = standardize internet penetration (% of population online) (0-100)
final_df["users_score"] = to_100(final_df["internet_pct"])

# digital_score = average of users_score and law_score (0-100)
final_df["digital_score"] = (final_df["users_score"] + final_df["law_score"]) / 2

# joint_score = average of privacy_score and digital_score (0-100)
final_df["joint_score"] = (final_df["privacy_score"] + final_df["digital_score"]) / 2

final_df.to_csv("final.csv", index=False)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Comparison
# ----------------------------------------------------------------------------------------------------------
final_df["privacy_rank"] = final_df["privacy_score"].rank(ascending=False).astype(int)
final_df["digital_rank"] = final_df["digital_score"].rank(ascending=False).astype(int)
final_df["joint_rank"] = final_df["joint_score"].rank(ascending=False).astype(int)

region_cols = ["africa", "latin_america_and_caribbean", "asia_and_oceania", "europe", "northern_america"]
final_df["region"] = final_df[region_cols].idxmax(axis=1)

# Comparison by Joint Rank
cols = ["country", "region", "privacy_score", "law_score", "users_score", "digital_score", "joint_score", "privacy_rank", "digital_rank", "joint_rank"]
joint_ranking_df = final_df.sort_values("joint_rank")[cols].round(2)
joint_ranking_df.to_csv("results/joint_ranking_df.csv", index=False)

# Region Counts
region_count_df = final_df["region"].value_counts()
region_count_df.to_csv("results/region_count_df.csv", index=True)

# Comparison by Region
five_num_summary = ["count", "min", "25%", "50%", "75%", "max"]

privacy_score_by_region_df = final_df.groupby("region")["privacy_score"].describe()[five_num_summary].round(2)
privacy_score_by_region_df.to_csv("results/privacy_score_by_region_df.csv", index=True)

digital_score_by_region_df = final_df.groupby("region")["digital_score"].describe()[five_num_summary].round(2)
digital_score_by_region_df.to_csv("results/digital_score_by_region_df.csv", index=True)

internet_pct_by_region_df = final_df.groupby("region")["internet_pct"].describe()[five_num_summary].round(2)
internet_pct_by_region_df.to_csv("results/internet_pct_by_region_df.csv", index=True)

overall_summary = final_df[["privacy_score", "digital_score", "internet_pct"]].describe().loc[five_num_summary].round(2)
overall_summary.to_csv("results/overall_summary.csv", index=True)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Figure 1: Boxplot of Comparitech Scores by Category
# ----------------------------------------------------------------------------------------------------------
score_cols = ["constitutional_protection", "statutory_protection", "privacy_enforcement", "identity_cards_and_biometrics", "data_sharing", 
              "visual_surveillance", "communication_interception", "workplace_monitoring", "government_access_to_data", "communication_data_retention",
              "surveillance_medical_financial_movement", "border_and_trans_border_issues", "leadership", "democratic_safeguards", "total"]
scores_long = final_df.melt(id_vars="country", value_vars=score_cols, var_name="score", value_name="value")
scores_long["score"] = scores_long["score"].str.replace("_", " ").str.title()
plt.figure(figsize=(12, 7))
sub_long = scores_long[scores_long["score"] != "Total"]
sns.boxplot(data=sub_long, y="score", x="value", color="cadetblue")
plt.title("Comparitech Privacy Scores by Category")
plt.xlabel("Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("results/visualizations/F1_comparitech_scores_by_category_boxplot.png", dpi=300)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Figure 2: Legislation Coverage by Region (Heatmap)
# ----------------------------------------------------------------------------------------------------------
order = ["All 5 laws", "Missing 1+ laws"]
has_law = (final_df[law_cols] == "Legislation") * 100
has_law["region"] = final_df["region"]
law_by_region = has_law.groupby("region").mean()
law_by_region.columns = law_by_region.columns.str.replace("_", " ").str.title()
law_by_region.index = law_by_region.index.str.replace("_", " ").str.title()

plt.figure(figsize=(13, 6))
sns.heatmap(law_by_region.T, 
            annot=True, 
            fmt=".0f", 
            cmap="Blues",
            vmin=0, 
            vmax=100, 
            cbar_kws={"label": "% of Countries with Legislation"})
plt.title("Legislation Coverage by Region")
plt.xlabel("")
plt.ylabel("")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("results/visualizations/F2_legislation_coverage_by_region_heatmap.png", dpi=300)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Figure 3: Number of Countries with Legislation Coverage by Region (Bar Chart)
# ----------------------------------------------------------------------------------------------------------
counts = pd.crosstab(final_df["region"], final_df["law_group"])[order]
counts.index = counts.index.str.replace("_", " ").str.title()
counts.plot(kind="bar", stacked=True, figsize=(9, 6), color=["steelblue", "forestgreen"])
plt.title("Number of Countries with Legislation Coverage by Region")
plt.xlabel("")
plt.ylabel("Number of Countries")
plt.xticks(rotation=0)
plt.legend(title="")
plt.tight_layout()
plt.savefig("results/visualizations/F3_number_of_countries_with_legislation_coverage_by_region.png", dpi=300)


# ----------------------------------------------------------------------------------------------------------
# Data Analysis - Figure 4: Percentage of Internet Users vs. Privacy Score (Scatter Plot)
# ----------------------------------------------------------------------------------------------------------
slope, intercept = np.polyfit(final_df["internet_pct"], final_df["total"], 1)
r = np.corrcoef(final_df["internet_pct"], final_df["total"])[0, 1]
x_line = np.linspace(final_df["internet_pct"].min(), final_df["internet_pct"].max(), 100)
final_df["region"] = final_df["region"].str.replace("_", " ").str.title()

fig, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(data=final_df, x="internet_pct", y="total", palette="Set1", hue="region", s=130, alpha=0.85, ax=ax)
ax.plot(x_line, intercept + slope * x_line, color="black", linewidth=2)
ax.set_title("Privacy Score vs Percentage of Internet Users", fontsize=16)
ax.set_xlabel("Percentage of Internet Users")
ax.set_ylabel("Comparitech Privacy Score")
ax.legend(fontsize=11, loc="lower left")
plt.tight_layout()
plt.savefig("results/visualizations/F4_internet_pct_vs_privacy.png", dpi=300)

plt.show()