import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from tqdm import tqdm

from capsnet import CapsNet
from functions import DigitMarginLoss
from functions import accuracy

batch_size = 32

train_loader = DataLoader(datasets.MNIST('data', train=True, download=False, transform=transforms.Compose([
    # transforms.RandomShift(2),
    transforms.ToTensor()])), batch_size=batch_size, shuffle=True)

test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([
    transforms.ToTensor()])), batch_size=batch_size)

model = CapsNet(with_reconstruction=False)
optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
margin_loss = DigitMarginLoss()
reconstruction_loss = torch.nn.MSELoss(size_average=False)
if torch.cuda.is_available():
    model.cuda()
    margin_loss.cuda()
    reconstruction_loss.cuda()

model.train()
for epoch in range(1, 11):
    epoch_tot_loss = 0
    epoch_tot_acc = 0
    bar = tqdm(train_loader, total=len(train_loader), initial=1)
    for data, target in bar:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)
        if model.with_reconstruction:
            digit_caps, reconstruction = model(data, target)
            loss = margin_loss(digit_caps, target) + 0.0005 * reconstruction_loss(reconstruction, data.view(-1))
        else:
            digit_caps = model(data, target)
            loss = margin_loss(digit_caps, target)

        epoch_tot_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        acc = accuracy(digit_caps, target)
        epoch_tot_acc += acc
        bar.set_description("epoch: {} [ loss: {:.4f} ] [ acc: {:.2f}% ]".format(epoch, epoch_tot_loss / batch_size,
                                                                                 100 * (epoch_tot_acc / batch_size)))
