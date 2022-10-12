import argparse
import os


class Option:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		self.parser.add_argument('--name', type=str, default='exp')
		self.parser.add_argument('--dataroot', required=True)
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
		self.parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'])

		self.parser.add_argument('--gpu_ids', type=str, default='0')
		self.parser.add_argument('--n_epochs', type=int, default=100)
		self.parser.add_argument('--n_epochs_decay', type=int, default=100)
		self.parser.add_argument('--batch_size', type=int, default=200)
		self.parser.add_argument('--lr', type=float, default=2e-4)
		self.parser.add_argument('--beta1', type=float, default=0.5)
		self.parser.add_argument('--num_threads', type=int, default=4)
		self.parser.add_argument('--pool_size', type=int, default=50, help="缓存池技巧")
		self.parser.add_argument('--lam', type=float, default=10.0, help="循环一致性损失权重")

		self.parser.add_argument('--print_loss_f', type=int, default=2, help="更新print_loss_f次网络打印一次损失")

	def print_opt(self, parser):
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(parser).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)  # 右左对齐
		message += '----------------- End -------------------'
		print(message)

		# 保存到本地
		expr_dir = os.path.join(parser.checkpoints_dir, parser.name)
		if not os.path.exists(expr_dir):
			os.makedirs(expr_dir)
		file_name = os.path.join(expr_dir, '{}_opt.txt'.format(parser.mode))
		with open(file_name, 'w') as f:
			f.write(message)

	def get_opt(self):
		parser = self.parser.parse_args()

		# 打印超参数
		self.print_opt(parser)

		# 将gpu_ids从str转为int
		str_ids = parser.gpu_ids.split(',')
		parser.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				parser.gpu_ids.append(id)

		return parser