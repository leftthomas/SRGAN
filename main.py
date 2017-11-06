import torch
import torchnet as tnt
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm

import config
import utils
from capsnet import CapsuleNet
from loss import CapsuleLoss


def processor(sample):
    data, labels, training = sample

    data = utils.augmentation(data.unsqueeze(1).float() / 255.0)
    labels = torch.LongTensor(labels)

    labels = torch.sparse.torch.eye(config.NUM_CLASSES).index_select(dim=0, index=labels)

    data = Variable(data)
    labels = Variable(labels)
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()

    if training:
        classes, reconstructions = model(data, labels)
    else:
        classes, reconstructions = model(data)

    loss = capsule_loss(data, labels, classes, reconstructions)

    return loss, classes


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()


def on_forward(state):
    meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])

    reset_meters()

    engine.test(processor, utils.get_iterator(False))
    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
    confusion_logger.log(confusion_meter.value())

    print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

    torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

    # reconstruction visualization

    test_sample = next(iter(utils.get_iterator(False)))

    ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
    if torch.cuda.is_available():
        _, reconstructions = model(Variable(ground_truth).cuda())
    else:
        _, reconstructions = model(Variable(ground_truth))
    reconstruction = reconstructions.cpu().view_as(ground_truth).data

    ground_truth_logger.log(
        make_grid(ground_truth, nrow=int(config.BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
    reconstruction_logger.log(
        make_grid(reconstruction, nrow=int(config.BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())


if __name__ == "__main__":
    model = CapsuleNet()
    if torch.cuda.is_available():
        model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(config.NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion Matrix',
                                                     'columnnames': list(range(config.NUM_CLASSES)),
                                                     'rownames': list(range(config.NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, utils.get_iterator(True), maxepoch=config.NUM_EPOCHS, optimizer=optimizer)
