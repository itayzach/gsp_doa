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
    def __init__(self, num_true_points_per_snr, num_false_points_per_snr, snr_vec):
        signals = generate_synthetic_data(num_true_points_per_snr, num_false_points_per_snr, snr_vec)

        self.total_num_true_points = snr_vec.size * num_true_points_per_snr
        self.total_num_false_points = snr_vec.size * num_false_points_per_snr
        self.num_true_points_per_snr = num_true_points_per_snr
        self.num_false_points_per_snr = num_false_points_per_snr
        self.snr_vec = snr_vec
        self.Xr = torch.tensor(np.real(signals['x']), dtype=torch.float)
        self.Xi = torch.tensor(np.imag(signals['x']), dtype=torch.float)
        self.label = torch.tensor(signals['label'], dtype=torch.long)
        self.signals = signals
        assert self.total_num_true_points + self.total_num_false_points == len(self.label)

    def __len__(self):
        return self.total_num_true_points + self.total_num_false_points

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
    plt.show()


def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def gnn_doa(model, test_set):
    model.eval()

    snr_vec = test_set.signals['snr']
    true_correct_vec = np.zeros(snr_vec.size)
    false_correct_vec = np.zeros(snr_vec.size)

    data_r = test_set.Xr
    data_i = test_set.Xi
    output = model(data_r, data_i)
    est_labels_vec = output.argmax(dim=1, keepdim=True)

    for k in range(test_set.__len__()):
        snr = test_set.signals['snr_rep'][k]
        ground_truth_label = test_set.signals['label'][k]
        ground_truth_theta = test_set.signals['theta'][k]

        snr_idx = np.argwhere(snr_vec == snr)
        est_label = est_labels_vec[k].item()
        if ground_truth_label == 1:
            true_correct_vec[snr_idx] += ground_truth_label == est_label
        elif ground_truth_label == 0:
            false_correct_vec[snr_idx] += ground_truth_label == est_label
        else:
            assert False, '?'

    true_accuracy_vs_snr = 100.0 * true_correct_vec / test_set.num_true_points_per_snr
    false_accuracy_vs_snr = 100.0 * false_correct_vec / test_set.num_false_points_per_snr

    return est_labels_vec, true_accuracy_vs_snr, false_accuracy_vs_snr


####################################################################################################################
# different model
####################################################################################################################
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        max_deg = 2
        A, Ar, Ai = get_adjacency(theta=theta_d)

        I = np.eye(A.shape[0])
        A = A + I
        dii = np.sum(A, axis=1, keepdims=False)
        D = np.diag(dii)
        D_inv_h = np.diag(dii ** (-0.5))
        # Laplacian
        L = np.matmul(D_inv_h, np.matmul(A, D_inv_h))

        self.gcn_layer1 = GCNLayer(L, N*M, N*M, max_deg)
        self.gcn_layer2 = GCNLayer(L, N*M, 10, max_deg)
        self.fc1 = nn.Linear(10, 2)

    def forward(self, xr, xi):
        xr, xi = self.gcn_layer1(xr, xi)
        xr, xi = self.gcn_layer2(xr, xi)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class GCNLayer(nn.Module):
    def __init__(self, L, in_features, out_features, max_deg=1):
        super(GCNLayer, self).__init__()
        self.fc_layers = []
        for i in range(max_deg):
            fc = ComplexLinear(in_features, out_features)
            self.add_module(f'fc_{i}', fc)
            self.fc_layers.append(fc)

        self.laplacians = self.calc_laplacian_functions(L, max_deg)

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

            # Batch size
            B = Xr.size(0)

            # AXr = Ar.mm(Xr) - Ai.mm(Xi)
            # AXi = Ar.mm(Xi) + Ai.mm(Xr)
            avg_neighbor_features_r = (torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xr.view(B, -1, 1)).view(B, -1)) - \
                                      (torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xi.view(B, -1, 1)).view(B, -1))
            avg_neighbor_features_i = (torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xi.view(B, -1, 1)).view(B, -1)) + \
                                      (torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xr.view(B, -1, 1)).view(B, -1))

            fc_r, fc_i = fc(avg_neighbor_features_r, avg_neighbor_features_i)
            Zr = fc_r + Zr
            Zi = fc_i + Zi

        Zr, Zi = complex_relu(Zr, Zi)
        return Zr, Zi