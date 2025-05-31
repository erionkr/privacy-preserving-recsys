# src/data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens 100k triples (user, item, rating).
    Expects a CSV with columns: rating, user, item, ...
    We ignore everything except 'user', 'item', and 'rating'.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.users = df["user"].values.astype("int64")
        self.items = df["item"].values.astype("int64")
        self.ratings = df["rating"].values.astype("float32")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.ratings[idx], dtype=torch.float32),
        )

def get_data_loaders(train_csv, test_csv, batch_size=256, num_workers=2):
    """
    Loads both train and test sets to determine num_users and num_items
    across the full index range of user/item.
    Returns train_loader, test_loader, num_users, num_items.
    """
    # 1) Read train and test CSVs only to determine unique counts:
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Union of all user IDs → total number of unique users
    all_users = pd.concat([df_train["user"], df_test["user"]]).unique()
    all_items = pd.concat([df_train["item"], df_test["item"]]).unique()

    num_users = len(all_users)
    num_items = len(all_items)

    # 2) Create Dataset objects for training/testing
    train_dataset = MovieLensDataset(train_csv)
    test_dataset = MovieLensDataset(test_csv)

    # 3) Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_users, num_items
