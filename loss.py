import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images).detach())
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = (((out_images[:, :, :-1, :] - out_images[:, :, 1:, :]) ** 2 + (
                out_images[:, :, :, :-1] - out_images[:, :, :, 1:]) ** 2) ** 1.25).mean()
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRTransformTest(object):
    def __init__(self, shrink_factor):
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.shrink_factor = shrink_factor

    def __call__(self, hr_tensor):
        hr_img = self.to_pil(hr_tensor)
        w, h = hr_img.size
        lr_scale = transforms.Scale(int(min(w, h) / self.shrink_factor), interpolation=3)
        hr_scale = transforms.Scale(min(w, h), interpolation=3)
        lr_img = lr_scale(hr_img)
        hr_restore_img = hr_scale(lr_img)
        return self.to_tensor(lr_img), self.to_tensor(hr_restore_img)


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
