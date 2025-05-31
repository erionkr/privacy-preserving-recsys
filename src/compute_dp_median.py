# src/compute_dp_median.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data_loader import get_data_loaders   # existing function from your project
from model import NCF                     # your NCF model class

def compute_gradient_norms(model, data_loader, device):
    """
    Iterate over each sample in the DataLoader (batch size = 1) and compute the L2 norm of the gradient.
    Returns a NumPy array containing all norms.
    """
    model.train()  # we need gradients
    norms = []
    loss_fn = nn.MSELoss()  # or the same loss used during training

    # We set batch size = 1 so that we obtain a single gradient vector per iteration.
    for batch in data_loader:
        user_ids, item_ids, ratings = batch
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        ratings = ratings.to(device)

        # 1. Forward
        preds = model(user_ids, item_ids)
        loss = loss_fn(preds.view(-1), ratings.float())

        # 2. Backward to obtain per-sample gradients
        model.zero_grad()
        loss.backward()

        # 3. Collect all parameter gradients into a single vector
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.detach().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        norms.append(total_norm)

        # Then zero out gradients again (the next model.zero_grad() will also do this)
        model.zero_grad()

    return np.array(norms)


def dp_median(real_median, eps_med, delta_med, N, L):
    """
    Applies the Laplace mechanism to the true median:
      - real_median: median of the collected norms
      - eps_med: privacy budget for the median query
      - delta_med: not directly used in Laplace, but included for documentation
      - N: number of samples, sensitivity ~ L / N
      - L: assumed upper bound (e.g., max(norms))
    """
    sensitivity = L / N
    scale = sensitivity / eps_med
    noise = np.random.laplace(loc=0.0, scale=scale)
    return float(real_median + noise)


def main():
    parser = argparse.ArgumentParser(description="Compute a DP-median clipping threshold")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="../Dataset/train.csv",
        help="Path to the training CSV (user-item ratings)"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="../Dataset/test.csv",
        help="Path to the test CSV (only to correctly determine num_items and num_users)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for gradient-norm calculation (ideal: 1)"
    )
    parser.add_argument(
        "--eps_med",
        type=float,
        default=0.5,
        help="Epsilon budget for the DP-median query"
    )
    parser.add_argument(
        "--delta_med",
        type=float,
        default=1e-5,
        help="Delta budget for the DP-median query (only for documentation)"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 32]
    )
    parser.add_argument(
        "--public_subset",
        type=float,
        default=1.0,
        help="If <1.0, uses a random subset of training data (e.g., 0.2 for 20%)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for DP-median: {device}")

    # 1. Load the training dataset in batch size = 1 mode (or a smaller subset)
    #    We use get_data_loaders, which returns train_loader, test_loader, num_users, num_items.
    #    We ignore test_loader here.
    train_loader_full, _, num_users, num_items = get_data_loaders(
        args.train_csv,
        args.test_csv,
        batch_size=args.batch_size,
        num_workers=0
    )

    # If we want to use only a subset of the data:
    if args.public_subset < 1.0:
        all_indices = list(range(len(train_loader_full.dataset)))
        subset_size = int(len(all_indices) * args.public_subset)
        sampled_indices = np.random.choice(all_indices, subset_size, replace=False).tolist()
        train_subset = Subset(train_loader_full.dataset, sampled_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = train_loader_full

    # 2. Initialize the same NCF model as used during training
    model = NCF(
        num_users,
        num_items,
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims
    ).to(device)

    model.load_state_dict(torch.load("../checkpoints/model_baseline.pt", map_location=device))
    model.train()  # we want to compute gradients, so set to train mode

    # 3. Compute per-sample gradient norms
    print("Computing per-sample gradient norms (could take some time)...")
    norms = compute_gradient_norms(model, train_loader, device)
    real_median = float(np.median(norms))
    L = float(np.max(norms))   # Upper bound on the norms, can be e.g. max(norms)
    N = len(norms)

    print(f"Real median of gradient norms: {real_median:.4f}")
    print(f"Max (L) of gradient norms: {L:.4f} (N = {N} samples)")

    # 4. DP-median query
    eps_med = args.eps_med
    delta_med = args.delta_med
    clipped_median = dp_median(real_median, eps_med, delta_med, N, L)
    print(f"DP-median (ε_med={eps_med}, δ_med={delta_med}): {clipped_median:.4f}")

    # 5. Save the clipped value to a text file
    output_path = "../Dataset/dp_clipping_value.txt"
    with open(output_path, "w") as f:
        f.write(f"{clipped_median:.6f}")
    print(f"DP clipping threshold saved to: {output_path}\n")


if __name__ == "__main__":
    main()
