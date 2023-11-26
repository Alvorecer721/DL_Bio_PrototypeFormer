import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class ProtoFormer(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        super(ProtoFormer, self).__init__(backbone, n_way, n_support)
        self.classifier_loss_fn = nn.CrossEntropyLoss()
        self.prototype_loss_fn = contrastive_loss
        self.pair_dist = None

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1)  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        # Compute sub-prototypes.
        k = self.n_support
        z_sub_support = z_support.repeat(1, k - 1, 1).view(self.n_way, k, k - 1, -1)
        z_sub_proto = z_sub_support.mean(2) # for now, later add protoformer

        # Compute pairwise distance between subprototypes
        self.pair_dist = euclidean_dist_3d(z_sub_proto)

        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        loss = self.classifier_loss_fn(scores, y_query) + self.prototype_loss_fn(self.pair_dist)
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

    mask = torch.eye(n)
    dist_sums = pairwise_dist.sum((2, 3))

    positive_sums = (dist_sums * mask).sum() + 1
    negative_sums = (dist_sums * (1 - mask)).sum() + 1

    return torch.exp(positive_sums / negative_sums / n)

