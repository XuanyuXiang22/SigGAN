from torch.optim import lr_scheduler


def get_scheduler(optim, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optim, lr_lambda=lambda_rule)
    return scheduler