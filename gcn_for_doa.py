from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import numpy.matlib
from scipy.spatial.distance import cdist
import scipy
from complexLayers import ComplexLinear
from complexFunctions import complex_relu
import matplotlib.pyplot as plt

# Parmeters
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

theta_d = 70.3  # [degrees]


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def getNoiselessSignal(theta):
    tau = delta * np.cos(theta * np.pi / 180) * fs / c  # [samples]
    d = np.exp(-1j * w0 / fs * (m - 1) * tau)  # steering vector

    # % % Signals
    # % after a bandpass filter around f0 (filter out the negative spectrum)
    x1_bp = B * np.exp(1j * w0 / fs * n)
    x_bp_M = np.matlib.repmat(np.transpose(np.column_stack(x1_bp)), 1, M)
    d_N = np.matlib.repmat(d, N, 1)
    x_nm = np.multiply(x_bp_M, d_N)
    x_noiseless = np.ravel(x_nm)

    return x_noiseless


def generateSyntheticData(K):
    SNR_vec = np.linspace(start=-20, stop=20, num=int(K / 2))

    X_true = np.zeros((int(K / 2), N * M), dtype=complex)
    # Generate training data of True
    for SNR_idx, SNR in enumerate(SNR_vec):
        # Signal
        sigma = B / 10 ** (SNR / 20)
        mu = 0
        awgn = sigma * np.random.normal(mu, sigma, N * M)

        x_noiseless = getNoiselessSignal(theta=70.3)
        X_true[SNR_idx, :] = x_noiseless + awgn

    X_false = np.zeros((int(K / 2), N * M), dtype=complex)
    for SNR_idx, SNR in enumerate(SNR_vec):
        theta = np.random.uniform(0, 180)  # [degrees]
        x_noisless = getNoiselessSignal(theta)

        # Signal
        sigma = B / 10 ** (SNR / 20)
        mu = 0
        awgn = sigma * np.random.normal(mu, sigma, N * M)

        X_false[SNR_idx, :] = x_noisless + awgn

    labels = np.concatenate((np.full(int(K / 2), True, dtype=int), np.full(int(K / 2), False, dtype=int)))
    X = np.concatenate((X_true, X_false), axis=0)

    labels = torch.tensor(labels, dtype=torch.long)
    Xr = torch.tensor(np.real(X), dtype=torch.float)
    Xi = torch.tensor(np.imag(X), dtype=torch.float)

    return Xr, Xi, labels


class GraphSignalsDataset(torch.utils.data.Dataset):
    def __init__(self, K):
        Xr, Xi, labels = generateSyntheticData(K)
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

        # XLr = Xr.mm(Lr) - Xi.mm(Li)
        # XLi = Xr.mm(Li) + Xi.mm(Lr)

        avg_neighbor_features_r = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1)) - \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1))
        avg_neighbor_features_i = (torch.bmm(self.Ar.unsqueeze(0).expand(B, -1, -1), xi.view(B, -1, 1)).view(B, -1)) + \
                                  (torch.bmm(self.Ai.unsqueeze(0).expand(B, -1, -1), xr.view(B, -1, 1)).view(B, -1))
        xr, xi = self.fc(avg_neighbor_features_r, avg_neighbor_features_i)

        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, ((data_r, data_i), target) in enumerate(train_loader):
        (data_r, data_i), target = (data_r.to(device), data_i.to(device)), target.to(device)
        optimizer.zero_grad()
        output = model(data_r, data_i)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data_r), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.item()))
            print(f'Train Epoch: {epoch} [{batch_idx * len(data_r)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct/len(train_loader.dataset)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data_r, data_i), target in test_loader:
            (data_r, data_i), target = (data_r.to(device), data_i.to(device)), target.to(device)
            output = model(data_r, data_i)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)

    # print(
    #     '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return correct/len(test_loader.dataset)


def plot_progress(args, train_acc_vec, test_acc_vec):
    plt.plot(max(test_acc_vec) * np.ones(len(test_acc_vec)), color='red', linestyle='dashed')
    plt.plot(train_acc_vec, label='training acc', linewidth=2)
    plt.plot(test_acc_vec, label=f'test acc ({max(test_acc_vec):.2f}%)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.xlim(0, args.epochs)
    plt.ylim(0, 100)


def main():
    plt.rcParams['font.size'] = 14
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=5,
                        help='input batch size for testing (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pred_edge', action='store_true', default=False,
                        help='predict edges instead of using predefined ones')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        GraphSignalsDataset(K=1000),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        GraphSignalsDataset(K=300),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = GraphNet().to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-1)
    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    train_acc_vec = np.array([])
    test_acc_vec = np.array([])

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        train_acc_vec = np.append(train_acc_vec, 100.0*train_acc)
        test_acc_vec = np.append(test_acc_vec, 100.0*test_acc)

    plot_progress(args, train_acc_vec, test_acc_vec)
    plt.show()


if __name__ == '__main__':
    main()
    # Examples:
    # python mnist_fc.py --model fc
    # python mnist_fc.py --model graph
    # python mnist_fc.py --model graph --pred_edge