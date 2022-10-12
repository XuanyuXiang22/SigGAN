import torch
import torch.nn as nn
from torch.nn import init


def init_net(net, gpu_ids):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			init.normal_(m.weight.data, 0.0, 0.2)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm') != -1:
			init.normal_(m.weight.data, 1.0, 0.2)
			init.constant_(m.bias.data, 0.0)

	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	net.apply(init_func)

	return net


class ResnetBlock(nn.Module):
		def __init__(self, input_nc):
			super(ResnetBlock, self).__init__()
			conv_block = []
			conv_block += [nn.ReflectionPad1d(1),
			               nn.Conv1d(input_nc, input_nc, kernel_size=3, padding=0, bias=True),
			               nn.InstanceNorm1d(input_nc),
			               nn.ReLU(True)]
			conv_block += [nn.ReflectionPad1d(1),
			               nn.Conv1d(input_nc, input_nc, kernel_size=3, padding=0, bias=True),
			               nn.InstanceNorm1d(input_nc)]
			self.conv_block = nn.Sequential(*conv_block)

		def forward(self, x):
			out = x + self.conv_block(x)
			return out


class Generator(nn.Module):
	def __init__(self, input_nc, ngf=64, n_blocks=9):
		super(Generator, self).__init__()

		model = [nn.ReflectionPad1d(3),
		         nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
		         nn.InstanceNorm1d(ngf),
		         nn.ReLU(True)]

		n_downsampling = 2  # 下采样次数
		for i in range(n_downsampling):
			mult = 2 ** i
			model += [nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
			          nn.InstanceNorm1d(ngf * mult * 2),
			          nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(n_blocks):  # add ResNet blocks
			model += [ResnetBlock(ngf * mult)]

		for i in range(n_downsampling):  # add upsampling layers
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2), kernel_size=3,
			                             stride=2, padding=1, output_padding=1, bias=True),
			          nn.InstanceNorm1d(int(ngf * mult / 2)),
			          nn.ReLU(True)]
		model += [nn.ReflectionPad1d(3)]
		model += [nn.Conv1d(ngf, 2, kernel_size=7, padding=0)]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)


class Discriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3):
		super(Discriminator, self).__init__()
		kw = 4
		padw = 1
		sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # gradually increase the number of filters
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
				nn.InstanceNorm1d(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True),
			nn.InstanceNorm1d(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [
			nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
		self.model = nn.Sequential(*sequence)

	def forward(self, x):
		return self.model(x)