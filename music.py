import pandas as pd

music_df = pd.read_csv("music.csv")

music_dummies = pd.get_dummies(music_df["genre"],drop_first=True )
print(music_dummies.head())

music_dummies = pd.concat([music_df, music_dummies], axis=1)

music_dummies = music_dummies.drop("genre", axis=1)