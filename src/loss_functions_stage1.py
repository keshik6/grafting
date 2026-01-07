import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true, clip_val=15):
        diff = y_pred - y_true
        diff = torch.clamp(diff, min=-clip_val, max=clip_val)
        cosh_diff = torch.cosh(diff)
        #max_cosh_diff = torch.clamp(cosh_diff, min=1e-12)  # Prevent log(0)
        log_cosh_diff = torch.log(cosh_diff)
        return torch.mean(log_cosh_diff)


def get_loss_fn_for_distillation(config):
    if config['loss'] == 'l2':
        return nn.MSELoss()
    elif config['loss'] == 'l1':
        return nn.L1Loss()
    elif config['loss'] == 'huber':
        return nn.HuberLoss(delta=config['delta'])
    else:
        print("Not implemented")