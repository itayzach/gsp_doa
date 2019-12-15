#!/usr/bin/env python

import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import classification_report
import scipy

ID_INSTR = 0
ID_ADMIN = 33


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


class GCNLayer(nn.Module):
    def __init__(self, graph_L, in_features, out_features, max_deg=1):
        super().__init__()

        self.fc_layers = []
        for i in range(max_deg):
            fc = nn.Linear(in_features, out_features, bias=(i == max_deg - 1))
            self.add_module(f'fc_{i}', fc)
            self.fc_layers.append(fc)

        self.laplacians = self.calc_laplacian_functions(graph_L, max_deg)

    def calc_laplacian_functions(self, L, max_deg):
        res = [L]
        for _ in range(max_deg - 1):
            res.append(torch.mm(res[-1], L))
        return res

    def forward(self, X):
        Z = torch.tensor(0.)
        for k, fc in enumerate(self.fc_layers):
            L = self.laplacians[k]
            LX = torch.mm(L, X)
            Z = fc(LX) + Z

        return torch.relu(Z)


def train_node_classifier(model, optimizer, X, y, epochs=60, print_every=10):
    y_pred_epochs = []
    for epoch in range(epochs+1):
        y_pred = model(X)
        y_pred_epochs.append(y_pred.detach())

        # Semi-supervised: only use labels of the Instructor and Admin nodes
        labelled_idx = [ID_ADMIN, ID_INSTR]
        loss = F.nll_loss(y_pred[labelled_idx], y[labelled_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f'Epoch {epoch:2d}, loss={loss.item():.5f}')
    return y_pred_epochs


def main():
    plt.rcParams['font.size'] = 14
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'running on: {device}')

    # % % Parmeters
    N = 41
    n = np.arange(0, N)    # samples
    M = 6
    m = np.arange(1, M+1)  # sensors
    f0 = 1e3               # singletone[Hz]
    w0 = 2 * np.pi * f0    # [rad / sec]
    fs = 8e3               # sampling rate[Hz]
    B = 1                  # amplitude[V]
    delta = 0.1            # uniform spacing between sensors[m]
    c = 340                # speed of sound[m / s]

    theta = 70.3  # [degrees]
    tau = delta * np.cos(theta * np.pi / 180) * fs / c  # [samples]
    d = np.exp(-1j * w0 / fs * (m - 1) * tau)  # steering vector

    # % % Signals
    # % after a bandpass filter around f0 (filter out the negative spectrum)
    x1_bp = B * np.exp(1j * w0 / fs * n)
    x_bp_M = np.matlib.repmat(np.transpose(np.column_stack(x1_bp)), 1, M)
    d_N = np.matlib.repmat(d, N, 1)
    x_nm = np.multiply(x_bp_M, d_N)
    x_noisless = np.ravel(x_nm)
    est_theta_vec = np.arange(0, 180)
    piquancy = np.zeros(len(est_theta_vec))

    SNR_vec = np.array([5, float("inf")])  # [dB]

    for SNR_idx, SNR in enumerate(SNR_vec):
        # Signal
        sigma = B / 10 ** (SNR / 20)
        mu = 0
        awgn = sigma * np.random.normal(mu, sigma, N * M)

        x = x_noisless + awgn
        print('SNR = {0:.2f}'.format(signaltonoise(abs(x))))
        for th_idx, est_theta in enumerate(est_theta_vec):

            # Adjacency matrix
            est_tau = delta * np.cos(est_theta * np.pi / 180) * fs / c  # [samples]

            a1_r = 0.5 * np.concatenate((np.array([0, np.exp(1j * w0 / fs * est_tau)]),
                                         np.zeros(M - 3),
                                         np.array([np.exp(1j * w0 / fs * (M - 1) * est_tau)])),
                                        axis=0)
            a1_c = a1_r.conj().T  # transpose and conj
            A1 = scipy.linalg.toeplitz(a1_c, a1_r)

            a2_r = 0.5 * np.concatenate((np.array([0, np.exp(-1j * w0 / fs)]),
                                         np.zeros(N - 3),
                                         np.array([np.exp(-1j * w0 / fs * (N - 1))])),
                                        axis=0)
            a2_c = a2_r.conj().T  # transpose and conj
            A2 = scipy.linalg.toeplitz(a2_c, a2_r)

            A = np.kron(A2, A1)

            if est_theta == theta:
                assert (np.linalg.norm(1 * x_noisless - np.matmul(A, x_noisless)) < 1e-9)

            G = nx.from_numpy_matrix(A)
            #G = nx.karate_club_graph()
            #ID_MEMBERS = set(G.nodes()) - {ID_ADMIN, ID_INSTR}

            # Visualize the Karate Club graph
            # fig, ax = plt.subplots(1,1, figsize=(14,8), dpi=100)
            # pos = nx.spring_layout(G)
            # cmap = cmap=plt.cm.tab10
            # node_colors = 0.4*np.ones(G.number_of_nodes())
            # node_colors[ID_INSTR] = 0.
            # node_colors[ID_ADMIN] = 1.
            # node_labels = {i: i for i in ID_MEMBERS}
            # node_labels.update({i: l for i,l in zip([ID_ADMIN, ID_INSTR],['A','I'])})
            # nx.draw_networkx(G, pos, node_color=node_colors, labels=node_labels, ax=ax, cmap=cmap)

            # Adjacency
            # A = nx.adj_matrix(G, weight=None)
            # A = np.array(A.todense())
            # Degree matrix
            dii = np.sum(A, axis=1, keepdims=False)
            D = np.diag(dii)
            # Laplacian
            L = D - A
            w, Phi = np.linalg.eigh(L)

            # Plot spectrum
            # plt.plot(w); plt.xlabel(r'$\lambda$')

            # Plot Fourier basis
            # fig, ax = plt.subplots(4, 4, figsize=(8,6), dpi=150)
            # ax = ax.reshape(-1)
            # vmin, vmax = np.min(Phi), np.max(Phi)
            # for i in range(len(ax)):
            #     nc = Phi[:,i]
            #     nx.draw_networkx(G, pos, node_color=nc, with_labels=False, node_size=15, ax=ax[i], width=0.4, cmap=plt.cm.magma, vmin=vmin, vmax=vmax)
            #     ax[i].axis('off')
            #     ax[i].set_title(rf'$\lambda_{{{i}}}={w[i]:.2f}$',fontdict=dict(fontsize=8))

            llambda, V = np.linalg.eigh(A)
            i_eig = np.argwhere(abs(llambda - 1) < 1e-10).ravel()
            i_eig = i_eig[0]


            x_hat = np.matmul(V.conj().T, x)
            # % figure; stem(abs(x_hat))
            x_hat_normed = x_hat / abs(x_hat[i_eig])
            x_hat_ = np.delete(x_hat_normed, i_eig)
            piquancy[th_idx] = 1 / np.sqrt(sum(abs(x_hat_) ** 2))
        piquancy = piquancy / max(piquancy)
        plt.axvline(x=theta, color='r')
        plt.plot(piquancy, label='SNR = ' + str(SNR) + ' [dB]', linewidth=2)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\xi(\theta)$')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 180, step=20))
    plt.xlim(est_theta_vec[0], est_theta_vec[-1])
    plt.ylim(0, 1)
    plt.show()

    assert(True, 'Continue from here...')
    # Input: features will be one-hot vectors (no actual info conveyed)
    X = torch.eye(G.number_of_nodes())

    # Create ground-truth labels
    labels = [(0 if d['club']=='Mr. Hi' else 1) for i,d in G.nodes().data()]
    labels = torch.tensor(labels, dtype=torch.long)

    # Labels represent group affiliation
    list(G.nodes().data())

    # Adjacency matrix
    A = nx.adj_matrix(G, weight=None)
    A = np.array(A.todense())
    I = np.eye(A.shape[0])
    A = A + I

    # Degree matrix
    dii = np.sum(A, axis=1, keepdims=False)
    D = np.diag(dii)

    # Normalized Laplacian
    D_inv_h = np.diag(dii**(-0.5))
    L = np.matmul(D_inv_h, np.matmul(A, D_inv_h))

    # ### Model
    torch.manual_seed(4)

    in_features, out_features = X.shape[1], 2
    graph_L = torch.tensor(L, dtype=torch.float)
    max_deg = 2
    hidden_dim = 10

    # Stack two GCN layers as our model
    gcn2 = nn.Sequential(
        GCNLayer(graph_L, in_features, hidden_dim, max_deg),
        GCNLayer(graph_L, hidden_dim, out_features, max_deg),
        nn.LogSoftmax(dim=1)
    )
    print(gcn2)

    # ### Training
    optimizer = torch.optim.Adam(gcn2.parameters(), lr=0.01)

    y_pred_epochs = train_node_classifier(gcn2, optimizer, X, labels)
    # Since our loss is calculated based on two samples only, it's not a good criterion of overall classification
    # accuracy.
    #
    # Let's look at the the accuracy over all nodes:
    y_pred = torch.argmax(gcn2(X), dim=1).numpy()
    y = labels.numpy()
    print(classification_report(y, y_pred, target_names=['I','A']))

    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main()