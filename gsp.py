from __future__ import print_function
import torch
import numpy as np
import numpy.matlib
import scipy
import matplotlib.pyplot as plt
from parameters import M, N, theta_d, delta, fs, c, w0, m, n, Amp


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


class Signal:
    def __init__(self, theta, SNR):
        self.SNR = SNR
        self.theta = theta
        self.label = (theta == theta_d)
        self.x_noiseless = self.get_noiseless_signal()
        self.awgn = self.gen_awgn()
        self.x = self.x_noiseless + self.awgn

    def get_noiseless_signal(self):
        tau = delta * np.cos(self.theta * np.pi / 180) * fs / c  # [samples]
        d = np.exp(-1j * w0 / fs * (m - 1) * tau)  # steering vector

        # after a bandpass filter around f0 (filter out the negative spectrum)
        x1_bp = Amp * np.exp(1j * w0 / fs * n)

        # xr_theta_d = np.real(x1_bp)
        # xi_theta_d = np.imag(x1_bp)
        # plt.figure()
        # plt.title(fr'Signal from theta_d')
        # plt.plot(xr_theta_d, label='Re{x(n)}')
        # plt.plot(xi_theta_d, label='Im{x(n)}')
        # plt.legend()
        # plt.show()

        x_bp_M = np.matlib.repmat(np.transpose(np.column_stack(x1_bp)), 1, M)
        d_N = np.matlib.repmat(d, N, 1)
        x_nm = np.multiply(x_bp_M, d_N)  # multiply by place (.* in matlab)
        x_noiseless = np.ravel(x_nm)  # matrix to vector (A(:) in matlab)

        return x_noiseless

    def gen_awgn(self):
        sigma = Amp / 10 ** (self.SNR / 20)
        mu = 0
        awgn = sigma * np.random.normal(mu, sigma, N * M)
        return awgn


def generate_synthetic_data(K):
    SNR_vec = np.linspace(start=-15, stop=100, num=int(K / 2))

    signals_true = []
    # Generate training data of True
    for SNR_idx, SNR in enumerate(SNR_vec):
        signals_true.append(Signal(theta_d, SNR))

    signals_false = []
    for SNR_idx, SNR in enumerate(SNR_vec):
        rand_theta = np.random.uniform(0, 180)  # [degrees]
        signals_false.append(Signal(rand_theta, SNR))

    signals = signals_true + signals_false

    labels = np.asarray([sig.label for sig in signals], dtype=np.long)
    X = np.asarray([sig.x for sig in signals])
    labels = torch.tensor(labels, dtype=torch.long)
    Xr = torch.tensor(np.real(X), dtype=torch.float)
    Xi = torch.tensor(np.imag(X), dtype=torch.float)

    return Xr, Xi, labels


def laplacian_doa():
    est_theta_vec = np.arange(0, 180)
    piquancy = np.zeros(len(est_theta_vec))

    SNR_vec = np.array([-15, float("inf")])  # [dB]
    plt.figure()
    for SNR_idx, SNR in enumerate(SNR_vec):
        # Signal
        sig = Signal(theta_d, SNR)
        x_noiseless = sig.x_noiseless
        x = sig.x
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

            if est_theta == theta_d:
                assert (np.linalg.norm(1 * x_noiseless - np.matmul(A, x_noiseless)) < 1e-9)

            # Degree matrix
            dii = np.sum(A, axis=1, keepdims=False)
            D = np.diag(dii)
            # Laplacian
            L = D - A
            w, Phi = np.linalg.eigh(L)

            # Plot spectrum
            if est_theta == theta_d:
                plt.figure()
                plt.plot(w)
                plt.xlabel(r'$\lambda$')

            llambda, V = np.linalg.eigh(A)
            i_eig = np.argwhere(abs(llambda - 1) < 1e-10).ravel()
            i_eig = i_eig[0]


            x_hat = np.matmul(V.conj().T, x)
            # % figure; stem(abs(x_hat))
            x_hat_normed = x_hat / abs(x_hat[i_eig])
            x_hat_ = np.delete(x_hat_normed, i_eig)
            piquancy[th_idx] = 1 / np.sqrt(sum(abs(x_hat_) ** 2))
        piquancy = piquancy / max(piquancy)

        plt.axvline(x=theta_d, color='r')
        plt.plot(piquancy, label='SNR = ' + str(SNR) + ' [dB]', linewidth=2)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\xi(\theta)$')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 180, step=20))
    plt.xlim(est_theta_vec[0], est_theta_vec[-1])
    plt.ylim(0, 1)


def plot_random_signals(data_loader):
    (xr_batch, xi_batch), y_batch = iter(data_loader).next()

    xr_theta_d = xr_batch[y_batch == 1, :]
    xi_theta_d = xi_batch[y_batch == 1, :]

    plt.figure()
    plt.title(fr'Signal from theta_d')
    plt.plot(xr_theta_d[0, 0:N - 1].numpy(), label='Re{x(n)}')
    plt.plot(xi_theta_d[0, 0:N - 1].numpy(), label='Im{x(n)}')
    plt.legend()

    xr_rand_theta = xr_batch[y_batch == 0, :]
    xi_rand_theta = xi_batch[y_batch == 0, :]

    plt.figure()
    plt.title(fr'Signal not from theta_d')
    plt.plot(xr_rand_theta[0, 0:N - 1].numpy(), label='Re{x(n)}')
    plt.plot(xi_rand_theta[0, 0:N - 1].numpy(), label='Im{x(n)}')
    plt.legend()