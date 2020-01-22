clc; clear; close all;

N = 64; n = (0:N-1)'; % samples
L = 2^nextpow2(N);
f0 = 1e3;             % singletone [Hz]
w0 = 2*pi*f0;         % [rad/sec]
fs = 8e3;             % sampling rate [Hz]
k = (-L/2:L/2-1);     % frequency bins
f_Hz = k*(fs/L);      % freq bins -> [Hz]
f_kHz = f_Hz/1e3;     % just in kHz
B = 1;                % amplitude [V]

%% Signals
x = B*cos(w0/fs*n);
figure; plot(x); title('cos');

x_hat = fftshift(fft(x, L))/L;

h_hat = zeros(L,1);
k0 = floor(L/2 + (f0/fs)*L + 1); % L/2 for the fftshift
h_hat(k0) = 1;

% filter in freq domain
y_hat = 2*x_hat.*h_hat;

y_filtered = ifft(fftshift(y_hat));
figure; stem(f_kHz, abs(y_hat)); xlabel('f [kHz]'); title('$\hat{y} = \hat{x} \hat{h}$', 'Interpreter', 'latex', 'FontSize', 14)

figure; stem(f_kHz, abs(x_hat)); xlabel('f [kHz]'); title('$\hat{x}$', 'Interpreter', 'latex', 'FontSize', 14)

y = B*exp(1j*w0/fs*n);

[y y_filtered(1:N) ]
