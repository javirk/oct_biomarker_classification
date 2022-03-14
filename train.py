import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever import OCTHDF5Dataset, Resize
import torchvision.transforms as transforms
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from random import randint
from time import sleep
import os
import copy
from libs import resnet


def main():
    # sleep(randint(1, 50))  # This is for the SLURM array jobs

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = Path(__file__).resolve().parents[0].joinpath('runs/DET_{}'.format(current_time))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)
    u.copy_file(FLAGS.config, f'{tb_path}/config.yml')

    train_path = data_path.joinpath('ambulatorium_all_slices.hdf5')
    validation_path = data_path.joinpath('oct_test_all.hdf5')

    if resize:
        t = transforms.Compose([Resize(resize), transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        t = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    totalset = OCTHDF5Dataset(train_path, transform_image=t)
    totalset_size = len(totalset)
    trainset_size = int(totalset_size * 0.9)
    testset_size = totalset_size - trainset_size
    trainset, testset = torch.utils.data.random_split(totalset, [trainset_size, testset_size])

    writing_freq_train = trainset_size // (writing_per_epoch * BATCH_SIZE)
    writing_freq_test = testset_size // BATCH_SIZE  # Only once per epoch

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    valset = OCTHDF5Dataset(validation_path, slice_set=None, transform_image=t)
    valset_size = len(valset)
    writing_freq_val = valset_size // BATCH_SIZE  # Only once per epoch
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = getattr(resnet, config['model_number'])(pretrained=config['from_pretrained'], num_classes=num_classes)

    print(f'GPU devices: {torch.cuda.device_count()}')
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6, momentum=0.9)
    lmbda = lambda epoch: 0.99
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    weights_loss = totalset.weights
    print(f'Applied weights for loss will be: {weights_loss.numpy()}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights_loss.to(device))

    best_roc = 0.0
    for epoch in range(N_EPOCHS):
        for phase in ['train', 'test', 'validation']:
            running_loss = 0.0
            running_pred = []
            running_true = []

            if phase == 'train':
                model.train()
                loader = trainloader
                writing_freq = writing_freq_train
                i_train = 0
            elif phase == 'test':
                model.eval()
                loader = testloader
                writing_freq = writing_freq_test
            elif phase == 'validation':
                model.eval()
                loader = valloader
                writing_freq = writing_freq_val

            for i, data in enumerate(loader):
                inputs, labels = data['images'].to(device).float(), data['labels'].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        i_train = i
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item()
                running_pred.append(outputs.sigmoid().detach().cpu().numpy())
                running_true.append(labels.detach().cpu().numpy())

                if i % writing_freq == (writing_freq - 1):
                    n_epoch = epoch * trainset_size // BATCH_SIZE + i_train + 1
                    epoch_loss = running_loss / (writing_freq * BATCH_SIZE)
                    dict_metrics = u.calculate_metrics_sigmoid(running_true, running_pred)
                    epoch_rocauc = dict_metrics['ROC AUC']
                    print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_rocauc}')
                    dict_metrics['Loss'] = epoch_loss
                    u.write_to_tb(writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                    running_pred = []
                    running_true = []
                    running_loss = 0.0

                    if phase == 'validation' and epoch_rocauc > best_roc:
                        best_roc = epoch_rocauc
                        best_model_wts = copy.deepcopy(model.state_dict())

            scheduler.step()
            print(f'Epoch {epoch} finished')
        torch.save(model.state_dict(),
                   Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_e{epoch}.pth'))

    torch.save(best_model_wts, Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_bestROC.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='configs/cfg_detector.yml',
                        type=str,
                        help='Path to the config file')

    parser.add_argument('-u', '--ubelix',
                        default=1,
                        type=int,
                        help='Running on ubelix (0 is no)')

    FLAGS, unparsed = parser.parse_known_args()
    config = u.read_config(FLAGS.config)

    model_type = config['model'].lower()
    BATCH_SIZE = config['batch_size']
    N_EPOCHS = config['epochs']
    LR = config['lr']
    writing_per_epoch = config['writing_per_epoch']
    pretrained_model = u.str2bool(config['from_pretrained'])
    try:
        resize = u.str2bool(config['resize'])
    except AttributeError:
        resize = config['resize']
    num_classes = config['classes']

    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parents[1].joinpath('Datasets')  # CHANGE THIS!
        num_workers = 0
    else:
        data_path = Path('/storage/homefs/jg20n729/OCT_Detection/Datasets')  # CHANGE THIS!
        num_workers = 8

    main()
