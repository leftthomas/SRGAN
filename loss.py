import torch
from torch import nn
from torchvision.models.vgg import vgg16


# Adversarial Loss with Content Loss
class GeneratorAdversarialWithContentLoss(nn.Module):
    def __init__(self):
        super(GeneratorAdversarialWithContentLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(torch.log(1 - out_labels))
        # Content Loss
        features_input = self.loss_network(out_images)
        features_target = self.loss_network(target_images)
        content_loss = self.mse_loss(features_input, features_target)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = (((out_images[:, :, :-1, :] - out_images[:, :, 1:, :]) ** 2 + (
            out_images[:, :, :, :-1] - out_images[:, :, :, 1:]) ** 2) ** 1.25).mean()

        return image_loss + 0.001 * adversarial_loss + 0.006 * content_loss + 2e-8 * tv_loss


if __name__ == "__main__":
    g_loss = GeneratorAdversarialWithContentLoss()
    print(g_loss)
