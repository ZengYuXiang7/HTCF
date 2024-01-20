# -*- coding: utf-8 -*-
# Author : yuxiang Zeng
import time

import dgl as d
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm

from utils.metamodel import MetaModel
from utils.trainer import get_optimizer
from utils.utils import to_cuda, optimizer_zero_grad, optimizer_step, lr_scheduler_step


class GATCF(MetaModel):
    def __init__(self, user_num, serv_num, args):
        super(GATCF, self).__init__(user_num, serv_num, args)
        self.args = args
        try:
            userg = pickle.load(open('./modules/models/baselines/userg.pk', 'rb'))
            servg = pickle.load(open('./modules/models/baselines/servg.pk', 'rb'))
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./modules/models/baselines/userg.pk', 'wb'))
            pickle.dump(servg, open('./modules/models/baselines/servg.pk', 'wb'))
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.user_embeds = torch.nn.Embedding(self.usergraph.number_of_nodes(), self.dim)
        torch.nn.init.kaiming_normal_(self.user_embeds.weight)
        self.item_embeds = torch.nn.Embedding(self.servgraph.number_of_nodes(), self.dim)
        torch.nn.init.kaiming_normal_(self.item_embeds.weight)
        self.user_attention = SpGAT(self.usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.item_attention = SpGAT(self.servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.cache = {}
        self.optimizer_embeds = get_optimizer(self.get_embeds_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.optimizer_tf = get_optimizer(self.get_attention_parameters(), lr=4e-3, decay=args.decay, args=args)
        self.optimizer_mlp = get_optimizer(self.get_mlp_parameters(), lr=1e-2, decay=args.decay, args=args)
        self.scheduler_tf = torch.optim.lr_scheduler.StepLR(self.optimizer_tf, step_size=args.lr_step, gamma=0.50)

    def forward(self, inputs, train):
        userIdx, itemIdx = inputs
        if train:
            Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
            user_embeds = self.user_embeds(Index)
            Index = torch.arange(self.servgraph.number_of_nodes()).cuda()
            serv_embeds = self.item_embeds(Index)

            user_embeds = self.user_attention(user_embeds)[userIdx]
            serv_embeds = self.item_attention(serv_embeds)[itemIdx]

            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        else:
            user_embeds = self.cache['user'][userIdx]
            serv_embeds = self.cache['serv'][itemIdx]
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        Index = torch.arange(self.usergraph.number_of_nodes()).cuda()
        user_embeds = self.user_embeds(Index)
        Index = torch.arange(self.servgraph.number_of_nodes()).cuda()
        serv_embeds = self.item_embeds(Index)
        user_embeds = self.user_attention(user_embeds)[torch.arange(339).cuda()]
        serv_embeds = self.item_attention(serv_embeds)[torch.arange(5825).cuda()]
        self.cache['user'] = user_embeds
        self.cache['serv'] = serv_embeds

    def get_embeds_parameters(self):
        parameters = []
        for params in self.user_embeds.parameters():
            parameters += [params]
        for params in self.item_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        for params in self.user_attention.parameters():
            parameters += [params]
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        for params in self.layers.parameters():
            parameters += [params]
        return parameters

    def train_one_epoch(self, dataModule):
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader, disable=not self.args.program_test):
            inputs, value = train_Batch
            if self.args.device == 'gpu':
                inputs, value = to_cuda(inputs, value)
            pred = self.forward(inputs, True)
            loss = self.loss_function(pred.to(torch.float32), value.to(torch.float32))
            optimizer_zero_grad(self.optimizer_embeds, self.optimizer_tf)
            optimizer_zero_grad(self.optimizer_mlp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.30)
            optimizer_step(self.optimizer_embeds, self.optimizer_tf)
            optimizer_step(self.optimizer_mlp)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        lr_scheduler_step(self.scheduler_tf)
        return loss, t2 - t1

class SpGAT(torch.nn.Module):
    def __init__(self, graph, nfeat, nhid, dropout, alpha, nheads, args):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.adj = self.get_adj_nrom_matrix(graph).cuda()
        self.numbers = len(self.adj)

        self.attentions = torch.nn.ModuleList()
        self.nheads = nheads

        for i in range(self.nheads):
            temp = SpGraphAttentionLayer(nfeat, nhid, dropout=args.dropout, alpha=alpha, concat=True)
            self.attentions += [temp]

        self.dropout_layer = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, embeds):
        x = self.dropout_layer(embeds)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, self.adj))
        return x

    @staticmethod
    def get_adj_nrom_matrix(graph):
        g = graph
        # 转换为邻接矩阵
        n = g.number_of_nodes()
        in_deg = g.in_degrees().numpy()
        rows = g.edges()[1].numpy()
        cols = g.edges()[0].numpy()
        adj = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

        def normalize_adj(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))  # 求每一行的和
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-0.5}
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # D^{-0.5}
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # adj = D^{-0.5}SD^{-0.5}, S=A+I
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj



class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(torch.nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(torch.nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.layer = torch.nn.Linear(in_features, out_features, bias=True)
        torch.nn.init.kaiming_normal_(self.layer.weight)

        self.a = torch.nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        torch.nn.init.kaiming_normal_(self.a.data)

        self.dropout = torch.nn.Dropout(dropout)
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

        self.norm = torch.nn.LayerNorm(out_features)

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = self.layer(input)
        # h: N x out
        assert not torch.isnan(input).any()
        assert not torch.isnan(self.layer.weight).any()
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2 * D x E


        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = self.norm(h_prime)

        # h_prime = h_prime.div(e_rowsum)
        # # h_prime: N x out
        # assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# GraphMF
def create_graph():
    userg = d.graph([])
    servg = d.graph([])
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/原始数据/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()
    slines = slines

    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)

    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)
    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)
    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:

    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        # 添加进入类别
        self.__category.add(category)
        # 如果类别不存在若无则，则新增一个类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则创建之，id += 1
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            # 如果值不存在与类别子树，则创建之
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    def query_value(self, id):
        # 返回值
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)


