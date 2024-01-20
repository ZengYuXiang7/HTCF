# -*- coding: utf-8 -*-
# Author : yuxiang Zeng


import time
import torch
from utils.metamodel import MetaModel

class HTCF(MetaModel):

    def __init__(self, user_num, serv_num, args):
        super().__init__(user_num, serv_num, args)
        self.user_num = user_num
        self.serv_num = serv_num
        self.dim = args.dimension

        self.user_embeds = torch.nn.Embedding(user_num, self.dim)
        self.serv_embeds = torch.nn.Embedding(serv_num, self.dim)

        self.hyper_user_embeds = torch.nn.Embedding(user_num, self.dim)
        self.hyper_serv_embeds = torch.nn.Embedding(serv_num, self.dim)

        self.mlp = torch.nn.Linear(self.dim * 3, self.dim)
        self.interaction = Interaction(args)

        # TODO:: get adj matrix
        self.adjmatrix = get_adj()

    def hgnnLayer(self, embeds, hyper):
        # HGNN can also be seen as learning a transformation in hidden space, with args.hyperNum hidden units (hyperedges)
        return embeds @ (hyper.T @ hyper)  # @ (embeds.T @ embeds)

    def pickRandomEdges(self, adj):
        edgeNum = adj._indices().shape[1]
        edgeSampNum = int(args.edgeSampRate * edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        rows = torch.randint(self.user_num, [edgeSampNum])
        cols = torch.randint(self.serv_num, [edgeSampNum])
        return rows, cols

    def forward(self, inputs, train=True):
        userIdx, servIdx = inputs
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.serv_embeds(servIdx)

        cat_embeds = torch.concat([user_embeds, serv_embeds], dim = 0)



        hyprer_embeds = (hyper_user_embeds * hyper_serv_embeds).sum(-1)
        embeds = (user_embeds * serv_embeds).sum(-1)


        #

        #
        hyprer_embeds = hyprer_embeds.sigmoid()
        embeds = embeds.sigmoid()
        #
        estimated = self.interaction(torch.concat([hyprer_embeds, embeds], dim = - 1))
        return estimated

    def prepare_test_model(self):
        pass


class Interaction(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        self.mlp = torch.nn.Linear(2 * args.dimension, args.dimension)

    def forward(self, inputs):
        inputs = self.mlp(inputs)
        outputs = torch.sum(inputs, dim = -1)
        return outputs
