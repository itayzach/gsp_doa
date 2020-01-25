from __future__ import print_function
import torch
import numpy as np
import numpy.matlib
import scipy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from parameters import M, N, theta_d, theta_threshold, theta_threshold_for_plot, delta, fs, c, w0, m, n, Amp, f0, delta_over_lambda, plots_dir


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def get_noiseless_signal(A, f, theta):
    tau = delta * np.cos(theta * np.pi / 180) * fs / c  # [samples]
    d = np.exp(-1j * (2*np.pi*f) / fs * (m - 1) * tau)  # steering vector

    # after a bandpass filter around f0 (filter out the negative spectrum)
    x1_bp = A * np.exp(1j * (2*np.pi*f) / fs * n)

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
    forbidden_idx = np.argwhere(abs(false_points - theta_d) < theta_threshold_for_plot).squeeze()
    while forbidden_idx.size > 0:
        false_points[forbidden_idx] = np.random.uniform(0, 180, forbidden_idx.size)
        forbidden_idx = np.argwhere(abs(false_points - theta_d) < theta_threshold_for_plot).squeeze()
    return false_points


def generate_synthetic_data(num_true_points_per_snr, num_false_points_per_snr, snr_vec, interferences):
    if delta > c/(2*f0):
        sign = '>'
    else:
        sign = '<'
    print(f'{delta} = delta ' + sign + f' lambda/2 = {c}/2x{f0/1e3}k = {c/(2*f0)}')

    total_num_true_points = snr_vec.size*num_true_points_per_snr
    total_num_false_points = snr_vec.size * num_false_points_per_snr
    K = total_num_true_points + total_num_false_points

    snr_rep_vec = np.concatenate((np.repeat(snr_vec, num_true_points_per_snr),
                                  np.repeat(snr_vec, num_false_points_per_snr)))

    not_theta_d_values = gen_not_theta_d_values(total_num_false_points)
    theta_vec = np.concatenate((theta_d*np.ones(total_num_true_points, dtype=float),
                                not_theta_d_values))
    noiseless_vec = np.empty([K, N*M], dtype=complex)
    x_hat_vec = np.empty([K, N*M], dtype=complex)
    x_vec = np.empty([K, N * M], dtype=complex)
    awgn_vec = np.empty([K, N*M], dtype=complex)
    labels_vec = np.empty(K, dtype=np.long)

    A, Ar, Ai = get_adjacency(theta=theta_d)
    llambda, V = np.linalg.eigh(A)

    for snr_idx, snr in enumerate(snr_rep_vec):
        theta = theta_vec[snr_idx]
        x_noiseless = get_noiseless_signal(A=Amp, f=f0, theta=theta)

        awgn = gen_awgn(snr)
        if interferences is not None:
            awgn += interferences
        x = x_noiseless + awgn
        label = abs(theta - theta_d) < theta_threshold

        x_hat = np.matmul(V.conj().T, x)

        # append
        noiseless_vec[snr_idx, :] = x_noiseless
        awgn_vec[snr_idx, :] = awgn
        x_vec[snr_idx, :] = x
        x_hat_vec[snr_idx, :] = x_hat
        labels_vec[snr_idx] = label

    signals = {
        'snr': snr_vec,
        'snr_rep': snr_rep_vec,
        'theta': theta_vec,
        'x_noiseless': noiseless_vec,
        'awgn': awgn_vec,
        'x': x_vec,
        'x_hat': x_hat_vec,
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
    theta_axis = np.linspace(start=0, stop=180, num=181*4)
    piquancy = np.zeros(len(theta_axis))
    num_points_per_snr = test_set.num_true_points_per_snr + test_set.num_false_points_per_snr

    # SNR_vec = np.array([-15, float("inf")])  # [dB]
    # plt.figure()

    plot_piquancy = True

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

        # # Degree matrix
        # dii = np.sum(A, axis=1, keepdims=False)
        # D = np.diag(dii)
        # # Laplacian
        # L = D - A
        # w, Phi = np.linalg.eigh(L)
        #
        # # Plot spectrum
        # if theta == theta_d:
        #     plt.figure()
        #     plt.plot(w)
        #     plt.xlabel(r'$\lambda$')

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
        est_theta_idx = np.argmax(piquancy)
        est_theta = theta_axis[est_theta_idx]
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

        if plot_piquancy and ground_truth_label == 1 and ground_truth_label != est_label and snr >= 16:
            fig = plt.figure()
            plt.plot(theta_axis[est_theta_idx - 50:est_theta_idx + 50], piquancy[est_theta_idx - 50:est_theta_idx + 50], label='SNR = ' + str(snr) + ' [dB]', linewidth=3)
            plt.axvline(x=theta_d, color='r', linewidth=3)
            plt.title(fr'$\theta_d = {theta_d:.1f}$; ' + r'$\hat{\theta}' + rf'= {est_theta:.2f}$')
            plt.xlabel(r'$\theta$')
            plt.ylabel(r'$\xi(\theta)$')
            plt.xlim(theta_axis[est_theta_idx - 50], theta_axis[est_theta_idx + 50])
            plt.ylim(0, 1)
            fig.savefig(plots_dir + '/piquancy_' + f'{theta_d:.1f}'.replace(".", "_") + '__' + f'{est_theta:.2f}'.replace(".", "_") + '.png', dpi=200)
            plot_piquancy = False
    true_accuracy_vs_snr = 100.0 * true_correct_vec / test_set.num_true_points_per_snr
    false_accuracy_vs_snr = 100.0 * false_correct_vec / test_set.num_false_points_per_snr

    return est_theta_vec, est_labels_vec, true_accuracy_vs_snr, false_accuracy_vs_snr


def plot_random_signal(signals, label, snr, interference_str,  theta_str, snr_str):
    idx = np.intersect1d(np.argwhere(signals['label'] == label), np.where(signals['snr_rep'] >= snr))
    assert len(idx) > 0, 'w00t?'
    if len(idx) > 1:
        idx = idx[0]

    x = signals['x'][idx]
    xr = np.real(x)
    xi = np.imag(x)
    x_hat = signals['x_hat'][idx]

    # split data to sensors
    indexes_1 = np.arange(start=0, stop=N * M, step=M)
    x1 = x[indexes_1]
    xr1 = xr[indexes_1]
    xi1 = xi[indexes_1]
    indexes_2 = np.arange(start=1, stop=N * M + 1, step=M)
    x2 = x[indexes_2]
    xr2 = xr[indexes_2]
    xi2 = xi[indexes_2]

    # fft
    L = 2 ** np.ceil(np.log2(N))  # nextpow2
    k = np.arange(-L / 2, L / 2)  # frequency bins
    f_Hz = k * (fs / L)           # freq bins -> [Hz]
    f_kHz = f_Hz / 1e3            # just in kHz

    x1_hat = np.fft.fftshift(np.fft.fft(x1, L))/L
    x2_hat = np.fft.fftshift(np.fft.fft(x2, L)) / L

    snr = signals['snr_rep'][idx]
    theta = signals['theta'][idx]

    # Plot
    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(top=0.18)
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.plot(xr1, label=r'Re{x_1(n)}')
    ax1.plot(xi1, label=r'Im{x_1(n)}')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.set_ylabel('Sensor #1')
    print(f'theta = {theta:.2f} with SNR = {snr:.2f} dB')

    ax2 = plt.subplot2grid((2, 3), (1, 0), sharex=ax1)
    ax2.plot(xr2, label=r'Re{x_1(n)}')
    ax2.plot(xi2, label=r'Im{x_1(n)}')
    ax2.set_xlabel('Time [Sample]')
    ax2.set_ylabel('Sensor #2')

    ax3 = plt.subplot2grid((2, 3), (0, 1))
    ax3.stem(f_kHz, np.abs(x1_hat), label=r'$|\hat{x}_1(f)|$')
    ax3.axes.get_yaxis().set_visible(False)
    ax3.axes.get_xaxis().set_visible(False)
    # ax3.set_ylabel(fr'$Y_1(f)$')

    ax4 = plt.subplot2grid((2, 3), (1, 1), sharex=ax3)
    ax4.stem(f_kHz, np.abs(x2_hat), label=r'$|\hat{x}_2(f)|$')
    ax4.axes.get_yaxis().set_visible(False)
    # ax3.set_ylabel(fr'$Y_2(f)$')
    ax4.set_xlabel('Frequency [KHz]')
    ax4.set_xticks(np.arange(f_kHz[0], f_kHz[-1]+1, step=2))

    ax5 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax5.stem(np.abs(x_hat))
    ellipse = Ellipse((x_hat.shape[0] - 1, np.abs(x_hat[-1])), width=12, height=1, edgecolor='r', fc='None', lw=2)
    ax5.add_patch(ellipse)
    ax5.axes.get_yaxis().set_visible(False)
    ax5.set_xlabel(r'$\lambda$')
    # ax5.set_ylabel(fr'$\hat{{y}}(\lambda)$')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.suptitle(fr'Signal from $\theta = {theta:.1f}^\circ$' + f' (label = {label}) ' + 'with ' + interference_str + ' interferences\n' +
                 f'with SNR = ${snr}$ [dB]; ' + fr'$\delta = {delta_over_lambda:.1f}\lambda$', y=0.98)
    fig.savefig(plots_dir + '/signal_' + theta_str + '_' + snr_str + '_' + interference_str + '_interferences' + '.png', dpi=200)
    # plt.show()
