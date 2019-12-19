from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import scipy
from complexLayers import ComplexLinear
from complexFunctions import complex_relu
import matplotlib.pyplot as plt
import networkx as nx
from parameters import M, N, theta_d, delta, fs, c, w0, plots_dir
from gsp import generate_synthetic_data


class GraphSignalsDataset(torch.utils.data.Dataset):
    def __init__(self, K):
        Xr, Xi, labels = generate_synthetic_data(K)
        self.K = K
        self.Xr = Xr
        self.Xi = Xi
        self.labels = labels

    def __len__(self):
        return self.K

    def __getitem__(self, index):
        return (self.Xr[index, :], self.Xi[index, :]), self.labels[index]


class GraphNet(nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.fc = ComplexLinear(N*M, 2)

        # precompute adjacency matrix before training
        # A = self.precompute_adjacency_images(img_size)
        Ar, Ai = self.get_adjacency()
        self.register_buffer('Ar', Ar)
        self.register_buffer('Ai', Ai)

    @staticmethod
    def get_adjacency():
        # Space-domain adjacency
        tau_d = delta * np.cos(theta_d * np.pi / 180) * fs / c  # [samples]
        a1_r = 0.5 * np.concatenate((np.array([0, np.exp(1j * w0 / fs * tau_d)]),
                                     np.zeros(M - 3),
                                     np.array([np.exp(1j * w0 / fs * (M - 1) * tau_d)])),
                                    axis=0)
        a1_c = a1_r.conj().T  # transpose and conj
        A1 = scipy.linalg.toeplitz(a1_c, a1_r)

        # Time-domain adjacency
        a2_r = 0.5 * np.concatenate((np.array([0, np.exp(-1j * w0 / fs)]),
                                     np.zeros(N - 3),
                                     np.array([np.exp(-1j * w0 / fs * (N - 1))])),
                                    axis=0)
        a2_c = a2_r.conj().T  # transpose and conj
        A2 = scipy.linalg.toeplitz(a2_c, a2_r)

        # Space-Time adjacency
        A = np.kron(A2, A1)

        print(A[:10, :10])

        Ar = torch.tensor(np.real(A), dtype=torch.float)
        Ai = torch.tensor(np.imag(A), dtype=torch.float)
        return Ar, Ai

    def forward(self, xr, xi):
        # Batch size
        B = xr.size(0)

        # AXr = Ar.mm(Xr) - Ai.mm(Xi)
        # AXi = Ar.mm(Xi) + Ai.mm(Xr)

        avg_neighbor_features_r = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1)) - \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1))
        avg_neighbor_features_i = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1)) + \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1))
        xr, xi = self.fc(avg_neighbor_features_r, avg_neighbor_features_i)

        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))

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
