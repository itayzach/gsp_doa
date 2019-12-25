from __future__ import print_function
import torch
import numpy as np
import numpy.matlib
import scipy
import matplotlib.pyplot as plt
from parameters import M, N, theta_d, theta_threshold, delta, fs, c, w0, m, n, Amp


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

    # x_nm is:
    #   -------> Space
    #   |     [ x(n=0,m=0),    ...,    x(n=0,m=M-1) ]
    #  Time   [    ...        x(n,m),      ...      ]
    #   |     [ x(n=N-1,m=0),  ...,  x(n=N-1,m=M-1) ]
    #  \/
    x_nm = np.multiply(x_bp_M, d_N)  # multiply by place (.* in matlab)

    # x_noiseless is row-stack of x_nm:
    # [  sensor0,     sensor1,  ...,   sensorM-1     |      sensor0,    ...,    sensorM-1   ]
    # [ x(n=0,m=0), x(n=0,m=1), ...,  x(n=0,m=M-1), ..., x(n=N-1,m=0),  ...,  x(n=N-1,m=M-1)]
    x_noiseless = np.ravel(x_nm)

    return x_noiseless


def gen_awgn(snr):
    sigma = Amp / 10 ** (snr / 20)
    mu = 0
    awgn = sigma * (np.random.normal(mu, sigma, N * M) + 1j*np.random.normal(mu, sigma, N * M))
    return awgn


def gen_not_theta_d_values(total_num_false_points):
    false_points = np.random.uniform(0, 180, total_num_false_points)
    forbidden_idx = np.argwhere(abs(false_points - theta_d) < theta_threshold).squeeze()
    while forbidden_idx.size > 0:
        false_points[forbidden_idx] = np.random.uniform(0, 180, forbidden_idx.size)
        forbidden_idx = np.argwhere(abs(false_points - theta_d) < theta_threshold).squeeze()
    return false_points


def generate_synthetic_data(num_true_points_per_snr, num_false_points_per_snr, snr_vec):
    total_num_true_points = snr_vec.size*num_true_points_per_snr
    total_num_false_points = snr_vec.size * num_false_points_per_snr
    K = total_num_true_points + total_num_false_points

    snr_rep_vec = np.concatenate((np.repeat(snr_vec, num_true_points_per_snr),
                                  np.repeat(snr_vec, num_false_points_per_snr)))

    not_theta_d_values = gen_not_theta_d_values(total_num_false_points)
    theta_vec = np.concatenate((theta_d*np.ones(total_num_true_points, dtype=float),
                                not_theta_d_values))
    noiseless_vec = np.empty([K, N*M], dtype=complex)
    x_vec = np.empty([K, N*M], dtype=complex)
    awgn_vec = np.empty([K, N*M], dtype=complex)
    labels_vec = np.empty(K, dtype=np.long)
    for snr_idx, snr in enumerate(snr_rep_vec):
        theta = theta_vec[snr_idx]
        x_noiseless = get_noiseless_signal(theta)

        awgn = gen_awgn(snr)
        x = x_noiseless + awgn
        label = abs(theta - theta_d) < theta_threshold

        # append
        noiseless_vec[snr_idx, :] = x_noiseless
        awgn_vec[snr_idx, :] = awgn
        x_vec[snr_idx, :] = x
        labels_vec[snr_idx] = label

    signals = {
        'snr': snr_vec,
        'snr_rep': snr_rep_vec,
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


def gsp_doa(test_set):
    theta_axis = np.arange(0, 180)
    piquancy = np.zeros(len(theta_axis))
    num_points_per_snr = test_set.num_true_points_per_snr + test_set.num_false_points_per_snr

    # SNR_vec = np.array([-15, float("inf")])  # [dB]
    # plt.figure()

    K = test_set.__len__()
    est_theta_vec = np.empty(K, dtype=np.long)
    est_labels_vec = np.empty(K, dtype=np.long)
    snr_vec = test_set.signals['snr']
    true_correct_vec = np.zeros(snr_vec.size, dtype=int)
    false_correct_vec = np.zeros(snr_vec.size, dtype=int)

    V_vec = []
    i_eig_vec = []
    print('Creating adjacency matrices...')
    for th_idx, theta in enumerate(theta_axis):
        # Adjacency matrix
        A, Ar, Ai = get_adjacency(theta)

        # Degree matrix
        dii = np.sum(A, axis=1, keepdims=False)
        D = np.diag(dii)
        # Laplacian
        L = D - A
        w, Phi = np.linalg.eigh(L)

        # Plot spectrum
        if theta == theta_d:
            plt.figure()
            plt.plot(w)
            plt.xlabel(r'$\lambda$')

        llambda, V = np.linalg.eigh(A)
        i_eig = np.argwhere(abs(llambda - 1) < 1e-10).ravel()
        i_eig = i_eig[0]

        V_vec.append(V)
        i_eig_vec.append(i_eig)

    print('Testing labels...')
    for k in range(K):
        # Signal
        x_noiseless = test_set.signals['x_noiseless'][k]
        snr = test_set.signals['snr_rep'][k]
        ground_truth_label = test_set.signals['label'][k]
        ground_truth_theta = test_set.signals['theta'][k]
        # awgn = gen_awgn(snr)
        x = test_set.signals['x'][k]
        # print('SNR = {0:.2f}'.format(signaltonoise(abs(x))))
        for th_idx, theta in enumerate(theta_axis):
            V = V_vec[th_idx]
            i_eig = i_eig_vec[th_idx]
            x_hat = np.matmul(V.conj().T, x)
            x_hat_normed = x_hat / abs(x_hat[i_eig])
            x_hat_ = np.delete(x_hat_normed, i_eig)
            piquancy[th_idx] = 1 / np.sqrt(sum(abs(x_hat_) ** 2))
        piquancy = piquancy / max(piquancy)
        est_theta = np.argmax(piquancy)
        est_theta_vec[k] = est_theta
        est_label = abs(est_theta - theta_d) < theta_threshold
        est_labels_vec[k] = est_label
        snr_idx = np.argwhere(snr_vec == snr)

        if ground_truth_label == 1:
            curr_points_per_snr = test_set.num_true_points_per_snr
            true_correct_vec[snr_idx] += ground_truth_label == est_label
            curr_correct_vec = true_correct_vec
        else:
            assert ground_truth_label == 0  # just making sure...
            curr_points_per_snr = test_set.num_false_points_per_snr
            false_correct_vec[snr_idx] += ground_truth_label == est_label
            curr_correct_vec = false_correct_vec

        if k % curr_points_per_snr == curr_points_per_snr - 1:
            curr_acc = 100. * curr_correct_vec[snr_idx].item() / curr_points_per_snr
            print(f'label = {ground_truth_label} | SNR = {snr} | ' +
                  f'accuracy = {curr_correct_vec[snr_idx].item()}/{curr_points_per_snr} ({curr_acc:.2f}%) | ' +
                  f'progress = {k}/{K} ({100. * k / K:.2f}%)')

        # plt.axvline(x=theta_d, color='r')
        # plt.plot(piquancy, label='SNR = ' + str(snr) + ' [dB]', linewidth=2)
        # plt.xlabel(r'$\theta$')
        # plt.ylabel(r'$\xi(\theta)$')
    # plt.legend(loc='upper right')
    # plt.xticks(np.arange(0, 180, step=20))
    # plt.xlim(theta_axis[0], theta_axis[-1])
    # plt.ylim(0, 1)
    # plt.show()

    true_accuracy_vs_snr = 100.0 * true_correct_vec / test_set.num_true_points_per_snr
    false_accuracy_vs_snr = 100.0 * false_correct_vec / test_set.num_false_points_per_snr

    return est_theta_vec, est_labels_vec, true_accuracy_vs_snr, false_accuracy_vs_snr


def plot_random_signal(signals, label, snr):
    idx = np.intersect1d(np.argwhere(signals['label'] == label), np.where(signals['snr_rep'] > snr))
    assert len(idx) > 0, 'w00t?'
    if len(idx) > 1:
        idx = idx[0]

    x = signals['x'][idx]
    xr = np.real(x)
    xi = np.imag(x)

    # split data to sensors
    indexes = np.arange(start=0, stop=N * M, step=M)
    x1 = x[indexes]
    xr1 = xr[indexes]
    xi1 = xi[indexes]

    # fft
    L = 2 ** np.ceil(np.log2(N))  # nextpow2
    k = np.arange(-L / 2, L / 2)  # frequency bins
    f_Hz = k * (fs / L)           # freq bins -> [Hz]
    f_kHz = f_Hz / 1e3            # just in kHz

    x1_hat = np.fft.fftshift(np.fft.fft(x1, L))/L
    snr = signals['snr_rep'][idx]
    theta = signals['theta'][idx]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title(fr'Signal from $\theta = {theta:.1f}^\circ$' + '\n' +
                  f'with SNR = ${snr:.2f}$ [dB]' + '\n' +
                  f'label = {label}')
    ax1.plot(xr1, label='Re{x(n)}')
    ax1.plot(xi1, label='Im{x(n)}')
    ax1.set_xlabel('n [sample]')
    ax1.legend()
    ax2.stem(f_kHz, np.abs(x1_hat), label=r'$|\hat{x}_1(f)|$')
    ax2.set_xlabel('f [kHz]')
    ax2.legend()
    plt.show()
