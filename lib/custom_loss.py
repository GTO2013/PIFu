import torch
from torch import nn

class CustomBCELoss(nn.Module):
    def __init__(self, brock=False, gamma=None):
        super(CustomBCELoss, self).__init__()
        self.brock = brock
        self.gamma = gamma

    def forward(self, pred, gt, gamma=None, w=None):
        x_hat = torch.clamp(pred, 1e-5, 1.0-1e-5) # prevent log(0) from happening
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        if self.brock:
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(gamma*x*torch.log(x_hat) + (1.0-gamma)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(gamma*gt*torch.log(x_hat) + (1.0-gamma)*(1.0-gt)*torch.log(1.0-x_hat))

        if w is not None:
            if len(w.size()) == 1:
                w = w[:,None,None]
            return (loss * w).mean()
        else:
            return loss.mean()

class CustomMSELoss(nn.Module):
    def __init__(self, gamma=None):
        super(CustomMSELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        weight = gamma * gt + (1.0-gamma) * (1 - gt)
        loss = (weight * (pred - gt).pow(2)).mean()

        if w is not None:
            return (loss * w).mean()
        else:
            return loss.mean()