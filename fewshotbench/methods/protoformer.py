from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class PEM(nn.Module):

    def __init__(self, x_dim, n_layer, n_head, ffn_dim, dropout=0.):
        super().__init__()

        layers = []
        in_dim = x_dim

        for _ in range(n_layer):
            layers.append(
                nn.TransformerEncoderLayer(d_model=in_dim, 
                                           nhead=n_head, 
                                           dim_feedforward=ffn_dim, 
                                           dropout=dropout, 
                                           activation='gelu', 
                                           norm_first=False, 
                                           batch_first=True
                                           )
            )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        prototype = x.mean(dim=-2, keepdim=True)
        x = torch.cat([prototype, x], dim=-2)

        x = self.encoder(x)
        return x

class ProtoFormer(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, n_sub_support, n_layer=1, n_head=2, ffn_dim=1280, contrastive_coef=1):
        super(ProtoFormer, self).__init__(backbone, n_way, n_support)
        self.classifier_loss_fn = nn.CrossEntropyLoss()
        self.prototype_loss_fn = contrastive_loss
        self.pair_dist = None
        self.pem = PEM(x_dim=self.feature.final_feat_dim, 
                       n_layer=n_layer, 
                       ffn_dim=ffn_dim,
                       n_head=n_head
                    )
        self.contrastive_coef = contrastive_coef
        self.n_sub_support = n_sub_support

    def set_forward(self, x):
        # Compute the prototypes (support) and queries (embeddings) for each datapoint.
        z_support, z_query = self.parse_feature(x, False)

        # Compute the prototype.
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_proto = self.pem(z_support)[:, 0]

        # Compute sub-prototypes.
        z_sub_support = self.generate_sub_supports(z_support, n_sub_supports=self.n_sub_support) # (n_way, n_combinations, n_sub_supports, **embedding_dim)
        n_combos, n_subs_with_token = z_sub_support.shape[1:3]
        z_sub_proto = self.pem(z_sub_support.view(self.n_way * n_combos, n_subs_with_token, -1))[:, 0].view(self.n_way, n_combos, -1)

        # Compute pairwise distance between subprototypes
        self.pair_dist = euclidean_dist_3d(z_sub_proto)

        # Format the queries for the similarity computation.
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Compute similarity score based on the euclidean distance between queries and prototypes.
        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        # Compute the similarity scores between the prototypes and the queries.
        scores = self.set_forward(x)

        # Create the category labels for the queries.
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        # y_query = Variable(y_query.cuda())

        # Compute the loss.
        loss = self.classifier_loss_fn(scores, y_query) + self.contrastive_coef * self.prototype_loss_fn(self.pair_dist)
        return loss
    
    def generate_sub_supports(self, support_set, n_sub_supports):
        """
        Generate sub-support sets for each category in the support set.
        
        :param support_set: Tensor of shape [n_way, n_support, **embedding_dim]
        :param n_sub_supports: Number of elements in each sub-support set
        :return: Tensor of shape [n_way, n_combinations, n_sub_supports, **embedding_dim]
                where n_combinations = n_support choose n_sub_supports
        """
        embedding_dim = support_set.size()[2:]
        combo_indices = list(combinations(range(self.n_support), n_sub_supports))
        n_combinations = len(combo_indices)
        sub_supports = torch.zeros(self.n_way, n_combinations, n_sub_supports, *embedding_dim, device=self.device)

        for way in range(self.n_way):
            for idx, combo in enumerate(combo_indices):
                sub_supports[way, idx] = support_set[way, list(combo)]

        return sub_supports

def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def euclidean_dist_3d(x):
    # dist[i][j] is matrix of pairwise distances of classes i and j
    # x: N x K x D
    n = x.size(0)
    k = x.size(1)
    d = x.size(2)

    flat_x = x.view(-1, d)
    dist = euclidean_dist(flat_x, flat_x)

    return dist.view(n, k, -1).transpose(1, 2).reshape(n , n, k, k)

def contrastive_loss(pairwise_dist):
    n = pairwise_dist.shape[0]

    # Get the device from the input tensor
    device = pairwise_dist.device

    # Create the mask tensor on the same device as pairwise_dist
    mask = torch.eye(n).to(device)
    
    dist_sums = pairwise_dist.sum((2, 3))

    positive_sums = (dist_sums * mask).sum() + 1
    negative_sums = (dist_sums * (1 - mask)).sum() + 1

    return torch.exp(positive_sums / negative_sums / n)