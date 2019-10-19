import pandas as pd
from sklearn import preprocessing


def set_printing_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)


def convert_yn_to_binary(value):
    if value == "y":
        return 1
    elif value == "n":
        return 0
    else:
        return None


def convert_rating_to_score(rating):
    ratings = {"excellent": 5, "good": 4, "ok": 3, "bad": 2}
    if rating in ratings:
        return ratings[rating]
    else:
        return 1


def get_age_bucket(age):
    if 6 <= age < 18:
        return "6-18"
    elif 18 <= age < 35:
        return "18-35"
    elif 35 <= age < 100:
        return "35-100"
    else:
        return "unknown"


set_printing_options()
conversions_df = pd.read_csv("data/ad_conversion.csv")

conversions_df["last name"].fillna("", inplace=True)

conversions_df.at[8, "gender"] = "F"
conversions_df.at[14, "gender"] = "M"
conversions_df.at[29, "gender"] = "F"
conversions_df.at[37, "gender"] = "M"

for z in range(len(conversions_df)):
    if conversions_df.at[z, "seen count"] > 1e6:
        conversions_df.at[z, "seen count"] = 0

conversions_df.insert(conversions_df.columns.get_loc("last name"), "full name", None)

for z in range(len(conversions_df)):
    conversions_df.at[z, "full name"] = (
            conversions_df.at[z, "first name"] + " " + conversions_df.at[z, "last name"]).strip()

conversions_df.drop(columns=["first name", "last name"], inplace=True)

conversions_df.insert(conversions_df.columns.get_loc("year of birth"), "birthday", None)

for z in range(len(conversions_df)):
    conversions_df.at[z, "birthday"] = pd.Timestamp(day=conversions_df.at[z, "day of birth"],
                                                    month=conversions_df.at[z, "month of birth"],
                                                    year=conversions_df.at[z, "year of birth"])

conversions_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)

for z in range(len(conversions_df)):
    conversions_df.at[z, "followed ad"] = convert_yn_to_binary(conversions_df.at[z, "followed ad"])
    conversions_df.at[z, "made purchase"] = convert_yn_to_binary(conversions_df.at[z, "made purchase"])
    conversions_df.at[z, "user rating"] = convert_rating_to_score(conversions_df.at[z, "user rating"])

conversions_df.insert(conversions_df.columns.get_loc("birthday"), "age", None)

for z in range(len(conversions_df)):
    conversions_df.at[z, "age"] = (pd.Timestamp.now() - conversions_df.at[z, "birthday"]).days // 365

conversions_df.drop(columns=["birthday"], inplace=True)

conversions_df.insert(conversions_df.columns.get_loc("age"), "age_bucket", None)

for z in range(len(conversions_df)):
    conversions_df.at[z, "age_bucket"] = get_age_bucket(conversions_df.at[z, "age"])

conversions_df.drop(columns=["age"], inplace=True)

conversions_df = conversions_df.astype({"followed ad": int, "made purchase": int, "user rating": int})

conversions_df.insert(conversions_df.columns.get_loc("user rating"), "ad effectiveness", None)

for z in range(len(conversions_df)):
    if conversions_df.at[z, "seen count"] > 0:
        conversions_df.at[z, "ad effectiveness"] = (conversions_df.at[z, "followed ad"] + conversions_df.at[
            z, "made purchase"]) / (2 * conversions_df.at[z, "seen count"])
    else:
        conversions_df.at[z, "ad effectiveness"] = -1

seen_count_scaler = preprocessing.MinMaxScaler()

conversions_df[["seen count"]] = seen_count_scaler.fit_transform(conversions_df[["seen count"]])

print(conversions_df)

conversions_df.to_csv("data/prepared_ad_conversion.csv")

colors_grouped = conversions_df[["color scheme", "followed ad", "made purchase"]].groupby("color scheme").mean()

print(colors_grouped)
