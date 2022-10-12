clc;
clear;
close all;
% (7, 4)hamming+BPSK+AWGN����������SNR�Ĺ�ϵ���߻���

k = 7;         % �볤
n = 4;         % ��Ϣλ��
n_group = 2000;  % ��Ϣλ������
SNR = (0:0.5:20);      % �����
SNR = SNR(:);
BER = zeros(length(SNR), 1);  % ������

for i_snr = 1:length(SNR)
    msg = randi([0 1], n * n_group, 1);   % �����͵�msg
    msg_hamming = zeros(k * n_group, 1);  % �ŵ�������msg
    msg_receive = zeros(n * n_group, 1);  % �ŵ�������msg
    % hamming encode
    for i = 1: n_group
        msg_hamming((i-1)*k+1:1:i*k) = encode(msg((i-1)*n+1:1:i*n), k, n, 'hamming/binary');
    end
    msg_BPSK = pskmod(msg_hamming, 2);      % BPSK����
    msg_AWGN = awgn(msg_BPSK, SNR(i_snr), 'measured');  % AWGN�ŵ�
    msg_demod = pskdemod(msg_AWGN, 2);      % BPSK���
    % hamming decode
    for i = 1: n_group
        msg_receive((i-1)*n+1:1:i*n) = decode(msg_demod((i-1)*k+1:1:i*k), k, n, 'hamming/binary');
    end
    % ������
    BER(i_snr) = biterr(msg, msg_receive) / n / n_group;
end

% ��ӡ��������������
semilogy(SNR, BER, '-r*')
xlabel('SNR(dB)')
ylabel('BER')
xlim([0 20])
ylim([1e-3 1])
set(gca, 'xTick', (0:2.5:20));
set(gca, 'XTicklabel', {'0.0', '2.5', '5.0', '7.5', '10.0', '12.5', '15.0', '17.5', '20.0'})
grid on