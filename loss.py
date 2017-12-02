import torch
from torch import nn
from torchvision.models.vgg import vgg19, vgg16


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


# vgg16 loss network
def vgg16_loss_network():
    vgg = vgg16(pretrained=True)
    relu = nn.Sequential(*(list(vgg.features.children())[:36])).eval()
    for param in relu.parameters():
        param.requires_grad = False
    return relu


# Adversarial Loss with Content Loss(default not with l1 loss)
class GeneratorAdversarialWithContentLoss(nn.Module):
    def __init__(self, loss_network, using_l1=False):
        super(GeneratorAdversarialWithContentLoss, self).__init__()
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.using_l1 = using_l1
        if self.using_l1:
            self.l1_loss = nn.L1Loss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(torch.log(1 - out_labels))
        # Content Loss
        features_input = self.loss_network(out_images)
        features_target = self.loss_network(target_images)
        content_loss = self.mse_loss(features_input, features_target)
        # image_loss = self.mse_loss(out_images, target_images)
        # g_tv_loss = (((out_images[:, :, :-1, :] - out_images[:, :, 1:, :]) ** 2 + (
        # out_images[:, :, :, :-1] - out_images[:, :, :, 1:]) ** 2) ** 1.25).mean()
        if self.using_l1:
            # L1 Loss
            l1_loss = self.l1_loss(out_images, target_images)
            return 1e-3 * adversarial_loss + content_loss + 1e-1 * l1_loss
        else:
            # return image_loss + 0.001 * adversarial_loss + 0.006 * content_loss + 2e-8 * g_tv_loss
            return 0.001 * adversarial_loss + 0.006 * content_loss


if __name__ == "__main__":
    g_loss = GeneratorAdversarialWithContentLoss(vgg16_loss_network())
    print(g_loss)
