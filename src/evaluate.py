# src/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from data_loader import get_data_loaders
from model import NCF

def evaluate_model(model, data_loader, device):
    model.eval()
    mse_criterion = nn.MSELoss(reduction="sum")
    mae_criterion = nn.L1Loss(reduction="sum")
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in data_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            preds = model(user_ids, item_ids)
            total_mse += mse_criterion(preds, ratings).item()
            total_mae += mae_criterion(preds, ratings).item()
            n += ratings.size(0)

    rmse = (total_mse / n) ** 0.5
    mae = total_mae / n
    return rmse, mae

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (we only need test_loader here, but get_data_loaders also returns train_loader)
    train_loader, test_loader, num_users, num_items = get_data_loaders(
        args.train_csv, args.test_csv, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Initialize the same model architecture
    model = NCF(num_users, num_items, embed_dim=args.embed_dim, hidden_dims=args.hidden_dims).to(device)

    # Load checkpoint (either baseline or DP)
    print(f"Loading checkpoint from {args.checkpoint_path}")
    # Load the raw state_dict
    raw_state = torch.load(args.checkpoint_path, map_location=device)

    # Remove the "_module." prefix from all keys
    stripped_state = {}
    for key, value in raw_state.items():
        # If the key starts with "_module.", remove exactly that prefix
        new_key = key.replace("_module.", "") 
        stripped_state[new_key] = value

    # Load into the model
    model.load_state_dict(stripped_state)

    # Evaluate on test set
    rmse, mae = evaluate_model(model, test_loader, device)
    print(f"Test RMSE: {rmse:.4f} | Test MAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved NCF model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--train_csv", type=str, default="../Dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="../Dataset/test.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
