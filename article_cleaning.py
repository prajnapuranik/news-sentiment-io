import pandas as pd


df = pd.read_csv("all_articles.csv")

df = df.drop(columns=["Date"])
df = df.dropna()

print(df.head())

df.to_csv("all_articles_cleaned.csv")