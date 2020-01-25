from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from complexLayers import ComplexLinear, ComplexConv1d
from complexFunctions import complex_relu
import matplotlib.pyplot as plt
import networkx as nx
from parameters import M, N, theta_d, delta, fs, c, w0, plots_dir
from gsp import generate_synthetic_data, get_adjacency


class GraphSignalsDataset(torch.utils.data.Dataset):
    def __init__(self, num_true_points_per_snr, num_false_points_per_snr, snr_vec, interferences):
        signals = generate_synthetic_data(num_true_points_per_snr, num_false_points_per_snr, snr_vec, interferences)

        self.total_num_true_points = snr_vec.size * num_true_points_per_snr
        self.total_num_false_points = snr_vec.size * num_false_points_per_snr
        self.num_true_points_per_snr = num_true_points_per_snr
        self.num_false_points_per_snr = num_false_points_per_snr
        self.snr_vec = snr_vec
        self.Xr = torch.tensor(np.real(signals['x']), dtype=torch.float).unsqueeze(2)
        self.Xi = torch.tensor(np.imag(signals['x']), dtype=torch.float).unsqueeze(2)
        # self.Xr = torch.tensor(np.real(signals['x_hat']), dtype=torch.float)
        # self.Xi = torch.tensor(np.imag(signals['x_hat']), dtype=torch.float)
        self.label = torch.tensor(signals['label'], dtype=torch.long)
        self.signals = signals
        assert self.total_num_true_points + self.total_num_false_points == len(self.label)

    def __len__(self):
        return self.total_num_true_points + self.total_num_false_points

    def __getitem__(self, index):
        return (self.Xr[index, :, :], self.Xi[index, :, :]), self.label[index]

    def get_signals(self):
        return self.signals


def plot_accuracy(args, model_name, train_acc_vec, test_acc_vec):
    fig = plt.figure()
    plt.plot(max(test_acc_vec) * np.ones(len(test_acc_vec)), linewidth=2, color='red', linestyle='dashed',
             label=f'Max test: {max(test_acc_vec):.2f}%')
    plt.plot(train_acc_vec, linewidth=2, label='Train')
    plt.plot(test_acc_vec, linewidth=2, label=f'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.title(model_name)
    plt.xlim(0, args.epochs-1)
    plt.ylim(80, 100)
    plt.tight_layout()
    fig.savefig(plots_dir + '/' + model_name + '_accuracy.png', dpi=200)
    # plt.show()


def visualize_complex_matrix(A, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.real(A), cmap='Blues')
    ax1.set_title('Real')
    ax2.imshow(np.imag(A), cmap='Blues')
    ax2.set_title('Imag')
    fig.suptitle(title)
    plt.show()


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
# GCN
####################################################################################################################
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        A, Ar, Ai = get_adjacency(theta=theta_d)

        # Add self loops
        I = np.eye(A.shape[0]) + 1j*np.eye(A.shape[0])
        A_h = A + I

        dii = np.sum(np.abs(A_h), axis=1, keepdims=False)
        D_inv_h = np.diag(dii ** (-0.5))

        # Laplacian
        L = np.matmul(D_inv_h, np.matmul(A_h, D_inv_h))
        # visualize_complex_matrix(L[0:10,0:10], 'L')

        max_deg = 2
        self.gcn_layer1 = GCNLayer(L, 1, 10, max_deg)
        self.gcn_layer2 = GCNLayer(L, 10, 1, max_deg)
        # self.fc1 = ComplexLinear(in_features=N*M, out_features=2, bias=True)
        self.fc1 = nn.Linear(in_features=N*M, out_features=2, bias=True)

    def forward(self, xr, xi):
        xr, xi = self.gcn_layer1(xr, xi)
        xr, xi = complex_relu(xr, xi)

        xr, xi = self.gcn_layer2(xr, xi)
        xr, xi = complex_relu(xr, xi)

        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        x = x.squeeze()
        x = self.fc1(x)
        x = F.relu(x)

        # xr = xr.squeeze()
        # xi = xi.squeeze()
        # xr, xi = self.fc1(xr, xi)
        # xr, xi = complex_relu(xr, xi)
        # x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))

        return F.log_softmax(x, dim=1)


class GCNLayer(nn.Module):
    def __init__(self, L, in_features, out_features, max_deg):
        super(GCNLayer, self).__init__()
        self.fc_layers = []
        for i in range(max_deg):
            fc = ComplexLinear(in_features, out_features, bias=True)
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

            # LXr = Lr.mm(Xr) - Li.mm(Xi)
            # LXi = Lr.mm(Xi) + Li.mm(Xr)
            if Xr.dim() == 2:
                LXr = (torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xr.view(B, -1, 1)).view(B, -1) - \
                      torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xi.view(B, -1, 1)).view(B, -1)).unsqueeze(2)
                LXi = (torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xi.view(B, -1, 1)).view(B, -1) + \
                      torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xr.view(B, -1, 1)).view(B, -1)).unsqueeze(2)

            else:
                LXr = torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xr) - \
                       torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xi)
                LXi = torch.bmm(Lr.unsqueeze(0).expand(B, -1, -1), Xi) + \
                       torch.bmm(Li.unsqueeze(0).expand(B, -1, -1), Xr)

            # fc_r, fc_i = fc(avg_neighbor_features_r, avg_neighbor_features_i)
            fc_r, fc_i = fc(LXr, LXi)
            Zr = fc_r + Zr
            Zi = fc_i + Zi

        # Zr, Zi = complex_relu(Zr, Zi)
        return Zr, Zi


####################################################################################################################
# CNN
####################################################################################################################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = ComplexConv1d(1, 20, bias=True, kernel_size=1)
        self.conv2 = ComplexConv1d(20, 1, bias=True, kernel_size=1)
        self.fc1 = nn.Linear(N * M, 2, bias=True)

    def forward(self, xr, xi):
        xr = xr.permute(0, 2, 1)
        xi = xi.permute(0, 2, 1)
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        x = x.squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)
