clc;
clear;
close all;
% ����BPSK��AWGN�µ��ŵ��ֲ�ѧϰ�ǳɶ�ѵ������������

n = 1000;  % ��������
len = 256;  % һ�����������ķ�����
variance = 1;  % AWGN�ķ���

rootDir='';
trainADir = fullfile(rootDir,'./trainA/');
trainBDir = fullfile(rootDir,'./trainB/');
if ~exist(trainADir,'dir')
    mkdir(trainADir);
end
if ~exist(trainBDir,'dir')
    mkdir(trainBDir);
end

for i = 1: n
    sig_mod = randi([0, 1], len, 1) * 2 - 1;
    % ��������
    sig_mod = transpose(sig_mod);
    pure_signal_real = real(sig_mod);
    pure_signal_imag = imag(sig_mod);
    save([trainADir, 'pure_signal_', num2str(i)], 'pure_signal_real', 'pure_signal_imag');
    txt = '%d trainA\n';
    fprintf(txt, i);
end

for i = 1: n
    sig_mod = randi([0, 1], len, 1) * 2 - 1;
    channel = normrnd(0, variance, len, 1);
    sig_cha = sig_mod + channel;
    % ��������
    sig_cha = transpose(sig_cha);
    channel_signal_real = real(sig_cha);
    channel_signal_imag = imag(sig_cha);
    save([trainBDir, 'channel_signal_', num2str(i)], 'channel_signal_real', 'channel_signal_imag');
    txt = '%d trainB\n';
    fprintf(txt, i);
end