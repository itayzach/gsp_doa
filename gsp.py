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


def get_noiseless_signal(theta):
    tau = delta * np.cos(theta * np.pi / 180) * fs / c  # [samples]
    d = np.exp(-1j * w0 / fs * (m - 1) * tau)  # steering vector

    # after a bandpass filter around f0 (filter out the negative spectrum)
    x1_bp = Amp * np.exp(1j * w0 / fs * n)

    x_bp_M = np.matlib.repmat(np.transpose(np.column_stack(x1_bp)), 1, M)
    d_N = np.matlib.repmat(d, N, 1)
    x_nm = np.multiply(x_bp_M, d_N)  # multiply by place (.* in matlab)
    x_noiseless = np.ravel(x_nm)  # matrix to vector (A(:) in matlab)

    return x_noiseless


def gen_awgn(snr):
    sigma = Amp / 10 ** (snr / 20)
    mu = 0
    awgn = sigma * (np.random.normal(mu, sigma, N * M) + 1j*np.random.normal(mu, sigma, N * M))
    return awgn


def generate_synthetic_data(K):
    snr_vec = np.concatenate((np.linspace(start=-15, stop=100, num=int(K / 2)), np.linspace(start=-15, stop=100, num=int(K / 2))))
    theta_vec = np.concatenate((theta_d*np.ones(int(K/2), dtype=float), np.random.uniform(0, 180, int(K/2))))
    noiseless_vec = np.empty([K, N*M], dtype=complex)
    x_vec = np.empty([K, N*M], dtype=complex)
    awgn_vec = np.empty([K, N*M], dtype=complex)
    labels_vec = np.empty(K, dtype=np.long)
    for snr_idx, snr in enumerate(snr_vec):
        theta = theta_vec[snr_idx]
        x_noiseless = get_noiseless_signal(theta)

        awgn = gen_awgn(snr)
        x = x_noiseless + awgn
        label = theta == theta_d

        # append
        noiseless_vec[snr_idx, :] = x_noiseless
        awgn_vec[snr_idx, :] = awgn
        x_vec[snr_idx, :] = x
        labels_vec[snr_idx] = label

    signals = {
        'snr': snr_vec,
        'theta': theta_vec,
        'x_noiseless': noiseless_vec,
        'awgn': awgn_vec,
        'x': x_vec,
        'label': labels_vec
    }

    return signals


def get_adjacency(theta):
    # Space-domain adjacency
    tau_d = delta * np.cos(theta * np.pi / 180) * fs / c  # [samples]
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

    Ar = torch.tensor(np.real(A), dtype=torch.float)
    Ai = torch.tensor(np.imag(A), dtype=torch.float)
    return A, Ar, Ai


def laplacian_doa():
    est_theta_vec = np.arange(0, 180)
    piquancy = np.zeros(len(est_theta_vec))

    SNR_vec = np.array([-15, float("inf")])  # [dB]
    plt.figure()
    for SNR_idx, snr in enumerate(SNR_vec):
        # Signal
        x_noiseless = get_noiseless_signal(theta_d)
        awgn = gen_awgn(snr)
        x = x_noiseless + awgn
        print('SNR = {0:.2f}'.format(signaltonoise(abs(x))))
        for th_idx, est_theta in enumerate(est_theta_vec):

            # Adjacency matrix
            A, Ar, Ai = get_adjacency(est_theta)

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
        plt.plot(piquancy, label='SNR = ' + str(snr) + ' [dB]', linewidth=2)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\xi(\theta)$')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 180, step=20))
    plt.xlim(est_theta_vec[0], est_theta_vec[-1])
    plt.ylim(0, 1)


def plot_random_signal(signals, label, snr):
    idx = np.intersect1d(np.argwhere(signals['label'] == label), np.where(signals['snr'] > snr))
    assert len(idx) > 0, 'w00t?'
    if len(idx) > 1:
        idx = idx[0]

    xr = np.real(signals['x'][idx])
    xi = np.imag(signals['x'][idx])
    x = signals['x'][idx]

    # split data to sensors
    x1 = x[0:N]
    xM = x[(M - 1) * N:M * N]

    # fft
    x1_h = np.abs(np.fft.fft(x1))/N
    xM_h = np.abs(np.fft.fft(xM))/N
    snr = signals['snr'][idx]
    theta = signals['theta'][idx]

    # Frequency axis
    f = 1e-3*(fs/2)/N * n

    # Plot
    plt.figure()
    plt.title(fr'Signal from $\theta = {theta:.1f}$' + '\n' + f'with SNR = ${snr:.2f}$ [dB]')
    plt.plot(xr[0:N - 1], label='Re{x(n)}')
    plt.plot(xi[0:N - 1], label='Im{x(n)}')
    # plt.plot(f, x1_h, label=r'$|\hat{x}_1(f)|$')
    # plt.plot(f, xM_h, label=r'$|\hat{x}_M(f)|$')
    plt.legend()
    plt.show()
