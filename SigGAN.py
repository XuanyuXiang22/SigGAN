import torch
import itertools
from network import Generator, init_net, Discriminator
from util.DataPool import DataPool
from Loss import GANLoss
from util.scheduler import get_scheduler


class SigGAN:
    def __init__(self, opt):
        super(SigGAN, self).__init__()
        self.opt = opt
        self.device = torch.device(self.opt.gpu_ids[0])
        self.optimizers = []

        self.netG_A = init_net(Generator(2), self.opt.gpu_ids)
        self.netG_B = init_net(Generator(2), self.opt.gpu_ids)

        if self.opt.mode == "train":
            self.netD_A = init_net(Discriminator(2), self.opt.gpu_ids)
            self.netD_B = init_net(Discriminator(2), self.opt.gpu_ids)
            # 缓存池trick
            self.fake_A_pool = DataPool(self.opt.pool_size)
            self.fake_B_pool = DataPool(self.opt.pool_size)
            # 损失函数
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            # 参数优化器
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # 学习率优化器
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

    def forward(self, x):
        self.real_A = x["A"].to(self.device)
        self.real_B = x["B"].to(self.device)
        self.image_paths = x["A_paths"]

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_G(self):
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lam
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lam
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def optimize_parameters(self, x):
        self.forward(x)  # 前向传播
        # 更新生成器参数
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # 更新判别器参数
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def train(self):
        """将所有网络设置为train模式"""
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def eval(self):
        """将所有网络设置为eval模式"""
        self.netG_A.eval()
        self.netG_B.eval()

    def update_lr(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
