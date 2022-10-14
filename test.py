import torch
from Option import Option
from SigDataset import get_loader
from SigGAN import SigGAN
from util.utils import load_networks, save_result


if __name__ == '__main__':
    opt = Option().get_opt()
    assert opt.mode == "test", "It is not test mode!"
    opt.batch_size = 1

    loader = get_loader(opt)
    model = SigGAN(opt)
    model.netG_A = load_networks(model.netG_A, opt)

    model.eval()
    with torch.no_grad():
        for x in loader:
            model.forward(x)  # 预测
            # 保存当前预测结果
            save_result(model.fake_B, x["A_paths"], opt)