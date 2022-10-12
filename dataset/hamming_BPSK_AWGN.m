clc;
clear;
close all;
% (7, 4)hamming+BPSK+AWGN，误码率与SNR的关系曲线绘制

k = 7;         % 码长
n = 4;         % 信息位长
n_group = 2000;  % 信息位的组数
SNR = (0:0.5:20);      % 信噪比
SNR = SNR(:);
BER = zeros(length(SNR), 1);  % 误码率

for i_snr = 1:length(SNR)
    msg = randi([0 1], n * n_group, 1);   % 待发送的msg
    msg_hamming = zeros(k * n_group, 1);  % 信道编码后的msg
    msg_receive = zeros(n * n_group, 1);  % 信道解码后的msg
    % hamming encode
    for i = 1: n_group
        msg_hamming((i-1)*k+1:1:i*k) = encode(msg((i-1)*n+1:1:i*n), k, n, 'hamming/binary');
    end
    msg_BPSK = pskmod(msg_hamming, 2);      % BPSK调制
    msg_AWGN = awgn(msg_BPSK, SNR(i_snr), 'measured');  % AWGN信道
    msg_demod = pskdemod(msg_AWGN, 2);      % BPSK解调
    % hamming decode
    for i = 1: n_group
        msg_receive((i-1)*n+1:1:i*n) = decode(msg_demod((i-1)*k+1:1:i*k), k, n, 'hamming/binary');
    end
    % 误码率
    BER(i_snr) = biterr(msg, msg_receive) / n / n_group;
end

% 打印对数误码率曲线
semilogy(SNR, BER, '-r*')
xlabel('SNR(dB)')
ylabel('BER')
xlim([0 20])
ylim([1e-3 1])
set(gca, 'xTick', (0:2.5:20));
set(gca, 'XTicklabel', {'0.0', '2.5', '5.0', '7.5', '10.0', '12.5', '15.0', '17.5', '20.0'})
grid on