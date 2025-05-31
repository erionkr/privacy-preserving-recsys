# src/train_dp.py

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from data_loader import get_data_loaders
from model import NCF
from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import os

def compute_dp_median_intern(model_proto, train_csv, test_csv,
                             batch_size, eps_med, delta_med,
                             embed_dim, hidden_dims, public_subset, device):
    """
    1) Loads training data with batch_size=1
    2) Computes the gradient norm per sample
    3) Calculates true median + max
    4) Applies Laplace noise to the median
    and returns the DP median.
    """

    # 1) Build DataLoader with batch_size=1
    train_loader_full, _, num_users, num_items = get_data_loaders(
        train_csv, test_csv,
        batch_size=batch_size,
        num_workers=0
    )

    if public_subset < 1.0:
        all_indices = list(range(len(train_loader_full.dataset)))
        subset_size = int(len(all_indices) * public_subset)
        sampled_indices = np.random.choice(all_indices, subset_size, replace=False).tolist()
        train_subset = Subset(train_loader_full.dataset, sampled_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = train_loader_full

    # 2) Initialize NCF model (unchanged)
    model_for_median = NCF(num_users, num_items,
                           embed_dim=embed_dim,
                           hidden_dims=hidden_dims).to(device)
    model_for_median.train()

    # 3) Compute gradient norm per sample
    loss_fn = nn.MSELoss()
    norms = []
    for user_ids, item_ids, ratings in train_loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        preds = model_for_median(user_ids, item_ids)
        loss = loss_fn(preds.view(-1), ratings)
        model_for_median.zero_grad()
        loss.backward()

        total_norm_sq = 0.0
        for p in model_for_median.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).item() ** 2
        l2_norm = total_norm_sq ** 0.5
        norms.append(l2_norm)
        model_for_median.zero_grad()

    norms = np.array(norms)
    real_median = float(np.median(norms))
    L = float(np.max(norms))
    N = len(norms)

    # 4) Apply DP Laplace mechanism to the median
    sensitivity = L / N
    scale = sensitivity / eps_med
    noise = np.random.laplace(0.0, scale)
    dp_median = real_median + noise
    return dp_median

def load_dp_clipping_value(path_to_txt: str) -> float:
    """
    Reads the DP median clipping result (a single float) from file.
    """
    if not os.path.exists(path_to_txt):
        raise FileNotFoundError(f"DP clipping file not found: {path_to_txt}")
    with open(path_to_txt, "r") as f:
        value = float(f.read().strip())
    return value

def train_epoch_dp(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for user_ids, item_ids, ratings in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        preds = model(user_ids, item_ids)
        loss = criterion(preds, ratings)
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
            loss = criterion(preds, ratings)
            total_loss += loss.item() * ratings.size(0)
    return total_loss / len(loader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load data (num_workers=0 to avoid Pickle error)
    train_loader, test_loader, num_users, num_items = get_data_loaders(
        args.train_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=0
    )

    # 2) Create NCF model
    model = NCF(num_users, num_items, embed_dim=args.embed_dim, hidden_dims=args.hidden_dims).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3) Determine clipping value
    if args.use_dp_median:
        clip_value = compute_dp_median_intern(
            model_proto=None,           # not used inside the function
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            batch_size=1,               # batch_size=1 for gradient norm computation
            eps_med=args.eps_med,
            delta_med=args.delta_med,
            embed_dim=args.embed_dim,
            hidden_dims=args.hidden_dims,
            public_subset=args.public_subset,
            device=device
        )
        print(f">>> Dynamically computed DP median (max_grad_norm) = {clip_value:.6f}")
    elif args.dp_clipping_path is not None:
        clip_value = load_dp_clipping_value(args.dp_clipping_path)
        print(f">>> Using DP clipping threshold from file: {clip_value:.6f}")
    else:
        clip_value = args.max_grad_norm
        print(f">>> Using manual max_grad_norm = {clip_value:.6f}")

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=clip_value,
    )

    best_test_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch_dp(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        # ε calculation (Privacy Accounting) using privacy_engine.get_epsilon()
        epsilon = privacy_engine.get_epsilon(delta=args.delta)
        print(
            f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | "
            f"Test MSE: {test_loss:.4f} | (ε = {epsilon:.2f}, δ = {args.delta})"
        )

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_dp.pt"))
            print(f"  → Saved new best DP model (MSE: {best_test_loss:.4f})")

    print("DP-SGD training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCF with DP-SGD (Opacus)")
    parser.add_argument("--train_csv", type=str, default="../Dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="../Dataset/test.csv")
    parser.add_argument("--output_dir", type=str, default="../checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier (σ)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (overwritten if --dp_clipping_path is set)")
    parser.add_argument("--dp_clipping_path", type=str, default=None, help="Path to file with DP median clipping value")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target δ for (ε, δ)-DP")
    parser.add_argument("--use_dp_median", action="store_true", help="If set: compute the clipping value dynamically with DP median.")
    parser.add_argument("--eps_med", type=float, default=0.5, help="Epsilon for DP median computation.")
    parser.add_argument("--delta_med", type=float, default=1e-5, help="Delta for DP median computation (for documentation).")
    parser.add_argument("--public_subset", type=float, default=1.0, help="Proportion of training data used for DP median (<1.0 enables subsampling).")
    args = parser.parse_args()
    main(args)
