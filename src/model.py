# src/model.py

import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    - Two embedding layers (user_embedding, item_embedding)
    - Then a stack of fully connected layers
    - Outputs a single scalar rating prediction
    """
    def __init__(self, num_users, num_items, embed_dim=32, hidden_dims=[64, 32]):
        super(NCF, self).__init__()
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        # MLP layers
        mlp_input_dim = embed_dim * 2
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(mlp_input_dim, dim))
            layers.append(nn.ReLU())
            mlp_input_dim = dim

        # Final output layer → single float
        layers.append(nn.Linear(mlp_input_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        """
        user_ids: LongTensor of shape (batch_size,)
        item_ids: LongTensor of shape (batch_size,)
        returns: Tensor of shape (batch_size, 1) – predicted rating
        """
        u_embed = self.user_embedding(user_ids)    # (batch, embed_dim)
        i_embed = self.item_embedding(item_ids)    # (batch, embed_dim)
        x = torch.cat([u_embed, i_embed], dim=-1)  # (batch, embed_dim*2)
        out = self.mlp(x)                          # (batch, 1)
        return out.squeeze(1)                     # (batch,)

