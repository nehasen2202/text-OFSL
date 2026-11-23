import torch
import torch.nn as nn
import numpy as np
from methods.meta_template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, verbose=False, adaptation=False, device='cuda:0'):
        super(ProtoNet, self).__init__(
            model_func=model_func,
            n_way=n_way,
            n_support=n_support,
            verbose=verbose,
            adaptation=adaptation,
            device=device
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x):
        z_support, z_query = self.parse_feature(x)

        z_proto = z_support.mean(1)  # [n_way, d]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)  # [n_way*n_query, d]

        dists = self.euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).to(self.device)
        scores = self.set_forward(x)
        return self.loss_fn(scores, y_query)
