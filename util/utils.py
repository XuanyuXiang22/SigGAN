"""
训练信息打印或可视化
"""
import os
import torch


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