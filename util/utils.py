"""
训练信息打印或可视化
"""
import os
import torch
import numpy as np
import scipy.io as scio


def print_loss(epoch, total_iters, losses, opt):
    msg = f"epoch: {epoch}, t_iter: {total_iters}, loss: "
    for k, v in losses.items():
        # tensor(device="gpu") -> float(device="cpu")
        v = v.detach().cpu().item()
        msg += f"{k}-{v:.3f}, "

    # 控制窗口打印
    print(msg)
    # 保存到check_points/name/train_loss.txt
    txt_file = os.path.join(opt.checkpoints_dir, opt.name, f"{opt.mode}_loss.txt")
    with open(txt_file, "a") as f:
        f.write('%s\n' % msg)


def save_networks(net, opt):
    save_name = 'G_A.pth'
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_name)

    if len(opt.gpu_ids) > 0:
        torch.save(net.module.cpu().state_dict(), save_path)  # 因为包了一层torch.nn.DataParallel
    else:
        torch.save(net.cpu().state_dict(), save_path)
    print("Save G_A done!")


def load_networks(net, opt):
    load_path = os.path.join(opt.checkpoints_dir, opt.name, "G_A.pth")
    state_dict = torch.load(load_path, map_location=opt.gpu_ids[0])
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    net = net.module  # 因为包了一层torch.nn.DataParallel
    net.load_state_dict(state_dict)
    net.to(opt.gpu_ids[0])
    net = torch.nn.DataParallel(net, opt.gpu_ids)

    return net


def save_result(data, path, opt):
    save_name = "fake_%s.mat" % path.split("_")[-1].split(".")[0]
    save_path = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # tensor gpu -> numpy cpu
    data = data.detach().cpu().numpy()
    scio.savemat(save_path + "/" + save_name, {'fake_real': data[0], 'fake_imag': data[1]})
    print("Test ", path, " done!")