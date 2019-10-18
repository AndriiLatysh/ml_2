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


def convert_rating_to_score(rating):
    ratings = {"excellent": 5, "good": 4, "ok": 3, "bad": 2}
    if rating in ratings:
        return ratings[rating]
    else:
        return 1


def get_bucket_for_age(age):
    if 6 <= age < 18:
        return "6-18"
    elif 18 <= age < 35:
        return "18-35"
    elif 35 <= age < 100:
        return "35-100"
    else:
        return "unknown"


set_printing_options()
conversion_df = pd.read_csv("data/ad_conversion.csv")

conversion_df["last name"].fillna("", inplace=True)

conversion_df.at[8, "gender"] = "F"
conversion_df.at[14, "gender"] = "M"
conversion_df.at[29, "gender"] = "F"
conversion_df.at[37, "gender"] = "M"

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] > 1e6:
        conversion_df.at[z, "seen count"] = 0

conversion_df.insert(conversion_df.columns.get_loc("last name"), "full name", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "full name"] = (
            conversion_df.at[z, "first name"] + " " + conversion_df.at[z, "last name"]).strip()

conversion_df.drop(columns=["first name", "last name"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("color scheme"), "birthday", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "birthday"] = pd.Timestamp(day=conversion_df.at[z, "day of birth"],
                                                   month=conversion_df.at[z, "month of birth"],
                                                   year=conversion_df.at[z, "year of birth"])

conversion_df.drop(columns=["day of birth", "month of birth", "year of birth"], inplace=True)

for z in range(len(conversion_df)):
    conversion_df.at[z, "followed ad"] = convert_yn_to_binary(conversion_df.at[z, "followed ad"])
    conversion_df.at[z, "made purchase"] = convert_yn_to_binary(conversion_df.at[z, "made purchase"])
    conversion_df.at[z, "user rating"] = convert_rating_to_score(conversion_df.at[z, "user rating"])

conversion_df.insert(conversion_df.columns.get_loc("birthday"), "age", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "age"] = (pd.Timestamp.now() - conversion_df.at[z, "birthday"]).days // 365

conversion_df.drop(columns=["birthday"], inplace=True)

conversion_df.insert(conversion_df.columns.get_loc("age"), "age group", None)

for z in range(len(conversion_df)):
    conversion_df.at[z, "age group"] = get_bucket_for_age(conversion_df.at[z, "age"])

conversion_df.drop(columns=["age"], inplace=True)

conversion_df = conversion_df.astype({"followed ad": int, "made purchase": int, "user rating": int})

conversion_df.insert(conversion_df.columns.get_loc("user rating"), "ad effectiveness", None)

for z in range(len(conversion_df)):
    if conversion_df.at[z, "seen count"] > 0:
        conversion_df.at[z, "ad effectiveness"] = (conversion_df.at[z, "followed ad"] + conversion_df.at[
            z, "made purchase"]) / (2 * conversion_df.at[z, "seen count"])
    else:
        conversion_df.at[z, "ad effectiveness"] = -1

seen_count_scaler = preprocessing.MinMaxScaler()
conversion_df[["seen count"]] = seen_count_scaler.fit_transform(conversion_df[["seen count"]])

conversion_df.to_csv("data/prepared_ad_conversions.csv")

colors_grouped = conversion_df[["color scheme", "followed ad", "made purchase"]].groupby("color scheme").mean()

print(colors_grouped)

print(conversion_df)
