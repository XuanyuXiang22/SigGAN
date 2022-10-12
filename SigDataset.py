from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import scipy.io as scio


class SigDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.mode + 'A')
        self.A_paths = self.makePaths(self.dir_A)

        if self.opt.mode == "train" or self.opt.mode == "val":
            self.dir_B = os.path.join(opt.dataroot, opt.mode + 'B')
            self.B_paths = self.makePaths(self.dir_B)

    def makePaths(self, dir):
        paths = os.listdir(dir)
        paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return paths

    def __getitem__(self, item):
        A_path = os.path.join(self.dir_A, self.A_paths[item])
        B_path = os.path.join(self.dir_B, self.B_paths[item])

        A_mat = scio.loadmat(A_path)
        A_real = A_mat['pure_signal_real']
        A_imag = A_mat['pure_signal_imag']
        A = np.vstack([A_real, A_imag])
        A = torch.from_numpy(A).float()

        if self.opt.mode == "train" or self.opt.mode == "val":
            B_mat = scio.loadmat(B_path)
            B_real = B_mat['channel_signal_real']
            B_imag = B_mat['channel_signal_imag']
            B = np.vstack([B_real, B_imag])
            B = torch.from_numpy(B).float()
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)


def get_loader(opt):
    # 获得dataset
    dataset = SigDataset(opt)
    # 获得dataloader
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False if opt.mode == "test" else True,
        num_workers=int(opt.num_threads),
    )
    return loader