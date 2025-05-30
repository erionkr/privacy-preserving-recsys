import pandas as pd
from sklearn.model_selection import train_test_split

# Files
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

# u.item => Movie titles & Genres 
items = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=list(range(24)),
)

genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

items.columns = ["item_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_cols

# Reducing to item_id, title and genres only
items = items[["item_id", "title"] + genre_cols]

# Reindexing user and item IDs to start from 0
user_ids = {uid: i for i, uid in enumerate(ratings["user_id"].unique())}
item_ids = {iid: j for j, iid in enumerate(ratings["item_id"].unique())}
ratings["user"] = ratings["user_id"].map(user_ids)
ratings["item"] = ratings["item_id"].map(item_ids)

# Merge ratings with items (for title & genres)
merged = ratings.merge(items, on="item_id", how="left")

# Drop unused columns
merged = merged.drop(columns=["user_id", "item_id", "timestamp"])

# Save dataset
merged.to_csv("full_dataset.csv", index=False)
print("full_dataset.csv saved!")

# Train/test
train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("Train.csv and test.csv saved!")
print("Preprocessing complete!")
