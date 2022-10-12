clc;
clear;
close all;
% BPSK_AWGN信道分布

n_symbol = 10000;  % 1/-1的个数
c_input = randi([0, 1], n_symbol, 1) * 2 - 1;
% 将c_input中的1和-1单独拿出来
c1 = c_input(c_input == 1);
c2 = c_input(c_input == -1);
% 分别通过AWGN信道
a1 = normrnd(0, 1, length(c1), 1);
a2 = normrnd(0, 1, length(c2), 1);
% 通过信道后的信号
o1 = c1 + a1;
o2 = c2 + a2;
% 可视化
x = (-4:0.1:4);
figure
hold on;
hist(o2, x);
hist(o1, x);
g = findobj(gca,'Type','patch');
set(g(1),'FaceColor',[1 0.5 0],'EdgeColor',[1 0.5 0])
set(g(2),'FaceColor',[0 0.5 0.5],'EdgeColor',[0 0.5 0.5])
set(gca, 'YTick', [100 200])
xlim([-4 4]);
ylim([0 230]);
xlabel('y');
ylabel('p(y|x)');
hold off;