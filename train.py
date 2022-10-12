from Option import Option
from SigDataset import get_loader
from SigGAN import SigGAN
from util.utils import print_loss, save_networks


if __name__ == '__main__':
    opt = Option().get_opt()   # 获得超参数
    loader = get_loader(opt)   # 获得dataloader
    model = SigGAN(opt)        # 获得网络模型，损失函数，参数更新器，学习率优化器

    total_iters = 0  # 更新网络参数的总次数
    model.train()
    for epoch in range(opt.n_epochs + opt.n_epochs_decay):
        # 用一个batch的数据更新一次网络参数
        for x in loader:
            model.optimize_parameters(x)
            total_iters += 1
            # 打印训练信息
            if total_iters % opt.print_loss_f == 0:
                losses = {"G_A": model.loss_G_A,
                          "G_B": model.loss_G_B,
                          "cycle_A": model.loss_cycle_A,
                          "cycle_B": model.loss_cycle_B,
                          "D_A": model.loss_D_A,
                          "D_B": model.loss_D_B}
                print_loss(epoch, total_iters, losses, opt)
        # 更新学习率
        model.update_lr()

    # 训练结束保存G_A网络参数
    save_networks(model.netG_A, opt)