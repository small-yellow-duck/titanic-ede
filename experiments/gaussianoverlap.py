import torch
import torch.nn

class GaussianOverlap(torch.nn.Module):
    """
    log loss based on overlap of two gaussians.

    """

    def __init__(self):
        super(GaussianOverlap, self).__init__()

    def forward(self, diffsquares, y, weights=None, do_batch_mean=True):
        if weights is None:
            weights = torch.ones_like(y)

        ln_area = torch.log(torch.clamp(1.0-torch.abs(torch.erf(torch.sqrt(torch.clamp(diffsquares, 0.0, None)) / 2.0)), 1e-8, 1.0))
        ln_1_min_area = torch.log(torch.clamp(torch.abs(torch.erf(torch.sqrt(torch.clamp(diffsquares, 0.0, None)) / 2.0)), 1e-8, 1.0))

        loss = -y*ln_area - (1-y)*ln_1_min_area
        if not do_batch_mean:
            return loss

        else:
            #print('loss shape, ', loss.size())
            loss = torch.sum(loss*weights)/torch.sum(weights)
            return loss