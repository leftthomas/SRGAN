import torch
from torch import nn
from torchvision.models.vgg import vgg19


# vgg19 loss network (default out with last relu feature map)
def vgg19_loss_network(is_last=True):
    vgg = vgg19(pretrained=True)
    if is_last:
        relu = nn.Sequential(*list(vgg.features)[:36])
    else:
        # second last
        relu = nn.Sequential(*list(vgg.features)[:27])
    for param in relu.parameters():
        param.requires_grad = False
    relu.eval()
    return relu


# only Adversarial Loss
class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super(GeneratorAdversarialLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()

    def forward(self, out_labels, target_labels, out_images=None, target_images=None):
        # Adversarial Loss
        adversarial_loss = self.adversarial_loss(out_labels, target_labels)
        return adversarial_loss


# Adversarial Loss with Pixel MSE Loss
class GeneratorAdversarialWithPixelMSELoss(nn.Module):
    def __init__(self):
        super(GeneratorAdversarialWithPixelMSELoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, target_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = self.adversarial_loss(out_labels, target_labels)
        # Pixel MSE Loss
        mse_loss = self.mse_loss(out_images, target_images)
        return 1e-3 * adversarial_loss + mse_loss


# Adversarial Loss with Content Loss(default not with l1 loss)
class GeneratorAdversarialWithContentLoss(nn.Module):
    def __init__(self, loss_network, using_l1=False):
        super(GeneratorAdversarialWithContentLoss, self).__init__()
        self.loss_network = loss_network
        self.adversarial_loss = nn.BCELoss()
        self.using_l1 = using_l1
        if self.using_l1:
            self.l1_loss = nn.L1Loss()

    def forward(self, out_labels, target_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = self.adversarial_loss(out_labels, target_labels)
        # Content Loss
        features_input = self.loss_network(out_images)
        features_target = self.loss_network(target_images)
        content_loss = torch.mean((features_input - features_target) ** 2)
        if self.using_l1:
            # L1 Loss
            l1_loss = self.l1_loss(out_images, target_images)
            return 1e-3 * adversarial_loss + content_loss + 1e-1 * l1_loss
        else:
            return 1e-3 * adversarial_loss + content_loss


if __name__ == "__main__":
    g_loss = GeneratorAdversarialLoss()
    print(g_loss)
