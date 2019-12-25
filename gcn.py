from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from complexLayers import ComplexLinear
from complexFunctions import complex_relu
import matplotlib.pyplot as plt
import networkx as nx
from parameters import M, N, theta_d, delta, fs, c, w0, plots_dir
from gsp import generate_synthetic_data, get_adjacency


class GraphSignalsDataset(torch.utils.data.Dataset):
    def __init__(self, K):
        signals = generate_synthetic_data(K)

        self.K = K
        self.Xr = torch.tensor(np.real(signals['x']), dtype=torch.float)
        self.Xi = torch.tensor(np.imag(signals['x']), dtype=torch.float)
        self.label = torch.tensor(signals['label'], dtype=torch.long)
        self.signals = signals

    def __len__(self):
        return self.K

    def __getitem__(self, index):
        return (self.Xr[index, :], self.Xi[index, :]), self.label[index]

    def get_signals(self):
        return self.signals


class GraphNet(nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.fc1 = ComplexLinear(N*M, 10)
        self.fc2 = nn.Linear(10, 2)

        # precompute adjacency matrix before training
        A, Ar, Ai = get_adjacency(theta=theta_d)
        self.register_buffer('Ar', Ar)
        self.register_buffer('Ai', Ai)

    def forward(self, xr, xi):
        # Batch size
        B = xr.size(0)

        # AXr = Ar.mm(Xr) - Ai.mm(Xi)
        # AXi = Ar.mm(Xi) + Ai.mm(Xr)
        avg_neighbor_features_r = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1)) - \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1))
        avg_neighbor_features_i = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1)) + \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1))
        xr, xi = self.fc1(avg_neighbor_features_r, avg_neighbor_features_i)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        x = F.relu(x)

        x = self.fc2(x)

        return x


def plot_accuracy(args, train_acc_vec, test_acc_vec):
    fig = plt.figure()
    plt.plot(max(test_acc_vec) * np.ones(len(test_acc_vec)), color='red', linestyle='dashed')
    plt.plot(train_acc_vec, label='training acc', linewidth=2)
    plt.plot(test_acc_vec, label=f'test acc ({max(test_acc_vec):.2f}%)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.xlim(0, args.epochs)
    plt.ylim(0, 100)
    fig.savefig(plots_dir + '/accuracy.png', dpi=200)


def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


####################################################################################################################
# different model
####################################################################################################################
class GNN(nn.Module):
    def __init__(self, L, in_features, hidden_dim, out_features, max_deg):
        super(GNN, self).__init__()
        self.gcn_layer1 = GCNLayer(L, in_features, hidden_dim, max_deg)
        self.gcn_layer2 = GCNLayer(L, hidden_dim, out_features, max_deg)

    def forward(self, xr, xi):
        xr, xi = self.gcn_layer1(xr, xi)
        xr, xi = self.gcn_layer2(xr, xi)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return F.log_softmax(x, dim=1)


class GCNLayer(nn.Module):
    def __init__(self, graph_L, in_features, out_features, max_deg=1):
        super(GCNLayer, self).__init__()
        self.fc_layers = []
        for i in range(max_deg):
            fc = ComplexLinear(in_features, out_features)
            self.add_module(f'fc_{i}', fc)
            self.fc_layers.append(fc)

        self.laplacians = self.calc_laplacian_functions(graph_L, max_deg)

    @staticmethod
    def calc_laplacian_functions(L, max_deg):
        res = [L]
        for _ in range(max_deg - 1):
            res.append(np.matmul(res[-1], L))
        return res

    def forward(self, Xr, Xi):
        Zr = torch.tensor(0., dtype=torch.float)
        Zi = torch.tensor(0., dtype=torch.float)
        for k, fc in enumerate(self.fc_layers):
            L = self.laplacians[k]
            Lr = torch.tensor(np.real(L), dtype=torch.float)
            Li = torch.tensor(np.imag(L), dtype=torch.float)

            XLr = Xr.mm(Lr) - Xi.mm(Li)
            XLi = Xr.mm(Li) + Xi.mm(Lr)

            fc_r, fc_i = fc(XLr, XLi)
            Zr = fc_r + Zr
            Zi = fc_i + Zi

        Zr, Zi = complex_relu(Zr, Zi)
        return Zr, Zi