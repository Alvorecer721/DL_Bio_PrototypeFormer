import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class PEM(nn.Module):

    def __init__(self, x_dim, layer_n, dropout=0.0):
        super().__init__()

        layers = []
        in_dim = x_dim

        for i in range(layer_n):
            layers.append(
                nn.TransformerEncoderLayer(d_model=in_dim, 
                                           nhead=2, 
                                           dim_feedforward=in_dim, 
                                           dropout=dropout, 
                                           activation='relu', 
                                           norm_first=True, 
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
    def __init__(self, backbone, n_way, n_support, layer_n=1, contrastive_coef=1):
        super(ProtoFormer, self).__init__(backbone, n_way, n_support)
        self.classifier_loss_fn = nn.CrossEntropyLoss()
        self.prototype_loss_fn = contrastive_loss_1
        self.pair_dist = None
        self.pem = PEM(self.feature.final_feat_dim, layer_n)
        self.contrastive_coef = contrastive_coef

    def set_forward(self, x):
        # Compute the prototypes (support) and queries (embeddings) for each datapoint.
        # Remember that you implemented a function to compute this before.
        z_support, z_query = self.parse_feature(x, False)

        # Compute the prototype.
        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        z_proto = self.pem(z_support)[:, 0]

        # Compute sub-prototypes.
        k = self.n_support
        z_sub_support = z_support.repeat(1, k - 1, 1).view(self.n_way * k, k - 1, -1)
        z_sub_proto = self.pem(z_sub_support)[:, 0].view(self.n_way, k, -1)

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
        y_query = Variable(y_query.cuda())

        # Compute the loss.
        loss = self.classifier_loss_fn(scores, y_query) + self.contrastive_coef * self.prototype_loss_fn(self.pair_dist)
        return loss

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

    mask = torch.eye(n).cuda()
    dist_sums = pairwise_dist.sum((2, 3))

    positive_sums = (dist_sums * mask).sum() + 1
    negative_sums = (dist_sums * (1 - mask)).sum() + 1

    return torch.exp(positive_sums / negative_sums / n)

def contrastive_loss_1(pairwise_dist):
    n = pairwise_dist.shape[0]

    mask = torch.eye(n).cuda()
    dist_sums = pairwise_dist.mean((2, 3))

    dist_sums_exp = torch.exp(dist_sums)
    positive_sums = torch.diagonal(dist_sums_exp)
    negative_sums = (dist_sums_exp * (1 - mask)).sum()

    return torch.log(positive_sums / negative_sums).mean()