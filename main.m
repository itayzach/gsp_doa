clc; clear; close all;

%% Parmeters
N = 41; n = (0:N-1)'; % samples
M = 6;  m = (1:M);    % sensors
f0 = 1e3;             % singletone [Hz]
w0 = 2*pi*f0;         % [rad/sec]
fs = 8e3;             % sampling rate [Hz]
B = 1;                % amplitude [V]
delta = 0.1;          % uniform spacing between sensors [m]
c = 340;              % speed of sound [m/s]

theta = 70.3;                       % [degrees]
tau = delta*cos(theta*pi/180)*fs/c; % [samples]
d = exp(-1j*w0/fs*(m-1)*tau);       % steering vector

%% Signals
% after a bandpass filter around f0 (filter out the negative spectrum)
x1_bp_noiseless = B*exp(1j*w0/fs*n);
SNR_vec = [inf 5];

est_theta_vec = 0:180;
% est_theta_vec = [theta 30 120];
piquancy = zeros(length(SNR_vec), length(est_theta_vec));
    
for SNR_idx = 1:length(SNR_vec)
    SNR = SNR_vec(SNR_idx);
    sigma = B/10^(SNR/20);
    awgn = sigma*randn(N,1);
    x1_bp = x1_bp_noiseless + awgn;
    disp(snr(x1_bp_noiseless, awgn))

    % build time-space signal matrix
    x_bp_M = repmat(x1_bp, 1, M);
    d_N = repmat(d, N, 1);
    x_nm = x_bp_M.*d_N;

    for th_idx = 1:length(est_theta_vec)
        %% Space-domain adjacency matrix
        est_theta = est_theta_vec(th_idx);
        est_tau = delta*cos(est_theta*pi/180)*fs/c; % [samples]

        a1_r = 0.5*[0 exp(1j*w0/fs*est_tau) zeros(1,M-3) exp(1j*w0/fs*(M-1)*est_tau)];
        a1_c = a1_r'; % transpose and conj
        A1 = toeplitz(a1_c, a1_r);

        % verify eigen vector with eigenvalue = 1
        if (est_theta == theta && SNR == Inf)
            for k = 1:N
                x_k = x_nm(k,:).';
                assert(norm(1*x_k - A1*x_k) < 1e-10);
            end
        end

        %% Time-domain adjacency matrix
        a2_r = 0.5*[0 exp(-1j*w0/fs) zeros(1,N-3) exp(-1j*w0/fs*(N-1))];
        a2_c = a2_r'; % transpose and conj
        A2 = toeplitz(a2_c, a2_r);

        % verify eigen vector with eigenvalue = 1
        if (est_theta == theta && SNR == Inf)
            for k = 1:M
                x_k = x_nm(:,k);
                assert(norm(1*x_k - A2*x_k) < 1e-10);
            end
        end

        %% Space-Time graph
        A = kron(A2, A1);
        x_mn = x_nm.';
        x = x_mn(:);
        if (est_theta == theta && SNR == Inf)
            assert(norm(1*x - A*x) < 1e-9);
        end

        %% GSP-toolbox
        % Gkron_coords = [(1:M*N)', zeros(M*N,1)];
        % Gkron_plotting_limits = [ ];
        % 
        % G_kron = gsp_graph(A); %, Gkron_coords, Gkron_plotting_limits);
        % G_kron = gsp_compute_fourier_basis(G_kron);

        % x_hat = G_kron.U' * x;


        %% GFT
        [V, Lambda] = eig(A);
        lambda = diag(Lambda);
        i_eig = find(abs(lambda - 1) < 1e-10);
        assert(length(i_eig) == 1, 'found more than one i_eig');
        x_hat = V' * x;
    %     figure; stem(abs(x_hat))
        x_hat_normed = x_hat/abs(x_hat(i_eig));
        x_hat_ = x_hat_normed([1:i_eig-1 i_eig+1:end]);
        piquancy(SNR_idx, th_idx) = 1/sqrt(sum(abs(x_hat_).^2));
    end
    piquancy(SNR_idx,:) = piquancy(SNR_idx,:)/max(piquancy(SNR_idx,:));
end
figure; 
plots = cell(size(SNR_vec));
for SNR_idx = 1:length(SNR_vec)
    plots{SNR_idx} = plot(est_theta_vec, piquancy(SNR_idx,:), 'LineWidth', 2, 'DisplayName', ['SNR = ' num2str(SNR_vec(SNR_idx)) ' [dB]']); 
    hold on;
end
xline(theta, '--r');
xlabel('$\theta$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\xi(\theta)$', 'Interpreter', 'latex', 'FontSize', 14);
legend([plots{1} plots{2}], 'Interpreter', 'latex', 'FontSize', 14, 'Location', 'best')
