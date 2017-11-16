from math import log10

import torch
from torchnet.meter import meter


class PSNRMeter(meter.Meter):
    def __init__(self):
        super(PSNRMeter, self).__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.sesum = 0.0

    def add(self, output, target):
        if not torch.is_tensor(output) and not torch.is_tensor(target):
            output = torch.from_numpy(output)
            target = torch.from_numpy(target)
        output = output.cpu()
        target = target.cpu()
        self.n += output.numel()
        self.sesum += torch.sum((output - target) ** 2)

    def value(self):
        mse = self.sesum / max(1, self.n)
        psnr = 10 * log10(1 / mse)
        return psnr
