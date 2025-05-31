# src/train_baseline.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import NCF
import argparse

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for user_ids, item_ids, ratings in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        preds = model(user_ids, item_ids)
        loss = criterion(preds.view(-1), ratings)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ratings.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            preds = model(user_ids, item_ids)
            loss = criterion(preds.view(-1), ratings)
            total_loss += loss.item() * ratings.size(0)
    return total_loss / len(loader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] Using device: {device}")

    # 1) load data
    train_loader, test_loader, num_users, num_items = get_data_loaders(
        args.train_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=0
    )

    # 2) generate model
    model = NCF(num_users, num_items, embed_dim=args.embed_dim, hidden_dims=args.hidden_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"[Baseline] Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

        # save best Model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_baseline.pt"))
            print(f"Saved new best baseline model (MSE: {best_test_loss:.4f})")

    print("Baseline training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCF without DP (Baseline)")
    parser.add_argument("--train_csv", type=str, default="../Dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="../Dataset/test.csv")
    parser.add_argument("--output_dir", type=str, default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32])
    args = parser.parse_args()
    main(args)
