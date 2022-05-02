import torch
import torch.nn as nn
import torch.optim as optim
from libs.data_retriever import OCTSlicesDataset, Resize
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import libs.utils as u
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
import argparse
from random import randint
from time import sleep
import os
import copy
import json
from libs import resnet

RNDM = 42
torch.manual_seed(RNDM)

def main():
    # sleep(randint(1, 50))  # This is for the SLURM array jobs

    os.makedirs('weights', exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_path = Path(__file__).resolve().parents[0].joinpath(
        f'runs/DETNEWLOSS_{current_time}_train{TRAIN_PERCENT:03d}_test{TEST_PERCENT:03d}_ep{N_EPOCHS:02d}_bs{BATCH_SIZE:03d}_lr{LR:.2E}_{TARGET}'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'dml'

    writer = SummaryWriter(tb_path)
    u.copy_file(FLAGS.config, f'{tb_path}/config.yml')

    csv_paths_train = [data_path.joinpath(f'annotation_{perc}_percent_export.csv') for perc in range(10, TRAIN_PERCENT + 1, 10)]    
    csv_paths_test = [data_path.joinpath(f'annotation_{perc}_percent_export.csv') for perc in range(10, TEST_PERCENT + 1, 10)]

    slices_path = data_path.joinpath('slices')

    if resize:
        t = transforms.Compose([Resize(resize), transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        t_test = transforms.Compose([Resize(resize), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
    else:
        t = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

    trainset = OCTSlicesDataset('train', csv_paths_train, slices_path, TARGET, transform_image=t)
    trainset_size = len(trainset)
    writing_freq_train = trainset_size // (writing_per_epoch * BATCH_SIZE)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testvalset = OCTSlicesDataset('test', csv_paths_test, slices_path, TARGET, transform_image=t_test)

    lbls_encoded = LabelEncoder().fit_transform([''.join(str(l)) for l in testvalset.label_set])
    sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RNDM)
    valset_indices, testset_indices = list(sgkf.split(testvalset.image_set, lbls_encoded, groups=testvalset.volume_set))[0]
    valset_size = len(valset_indices)
    testset_size = len(testset_indices)

    valset = torch.utils.data.Subset(testvalset, valset_indices)
    testset = torch.utils.data.Subset(testvalset, testset_indices)

    writing_freq_val = valset_size // BATCH_SIZE  # Only once per epoch
    writing_freq_test = testset_size // BATCH_SIZE  # Only once per epoch

    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    pd.DataFrame(
        zip(testvalset.df.iloc[testset_indices].sum(), testvalset.df.iloc[valset_indices].sum()), 
        columns=['test', 'val'], 
        index=testvalset.df.columns
    ).to_csv(tb_path / 'val_test_distribution.csv')

    model = getattr(resnet, config['model_number'])(pretrained=config['from_pretrained'], num_classes=num_classes)

    print(f'GPU devices: {torch.cuda.device_count()}')
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    print(trainset_size, valset_size, testset_size)

    model.to(device)

    if FLAGS.mode == 'train':

        model.train()

        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6, momentum=0.9)
        lmbda = lambda epoch: 0.99
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

        print(f'Applied weights for training loss will be: {trainset.weights.numpy()}')

        criterion_train = nn.BCEWithLogitsLoss(pos_weight=trainset.weights.to(device)) 
        criterion_val   = nn.BCEWithLogitsLoss(pos_weight=valset.dataset.weights.to(device)) 
        criterion_test  = nn.BCEWithLogitsLoss(pos_weight=testset.dataset.weights.to(device)) 

        best_rocauc_samples = 0.0
        best_rocauc_weighted = 0.0
        
        for epoch in range(N_EPOCHS):

            for phase in ['train', 'test', 'validation']:
                running_loss = 0.0
                running_pred = []
                running_true = []

                if phase == 'train':
                    model.train()
                    loader = trainloader
                    writing_freq = writing_freq_train
                    criterion = criterion_train
                    i_train = 0
                elif phase == 'validation':
                    model.eval()
                    loader = valloader
                    writing_freq = writing_freq_val
                    criterion = criterion_val
                elif phase == 'test':
                    model.eval()
                    loader = testloader
                    writing_freq = writing_freq_test
                    criterion = criterion_test

                for i, data in enumerate(loader):
                    inputs, labels = data['images'].to(device).float(), data['labels'].to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) # sigmoid is included in BCEWithLogitsLoss

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
                        epoch_rocauc_samples = dict_metrics['ROC AUC samples']
                        epoch_rocauc_weighted = dict_metrics['ROC AUC weighted']
                        print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_rocauc_samples}')
                        dict_metrics['Loss'] = epoch_loss
                        u.write_to_tb(writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                        running_pred = []
                        running_true = []
                        running_loss = 0.0

                        if phase == 'validation' and epoch_rocauc_samples > best_rocauc_samples: 
                            best_rocauc_samples = epoch_rocauc_samples
                            # best_model_wts = copy.deepcopy(model.state_dict())
                            best_model_samples = copy.deepcopy(model) 

                        if phase == 'validation' and epoch_rocauc_weighted > best_rocauc_weighted: 
                            best_rocauc_weighted = epoch_rocauc_weighted
                            # best_model_wts = copy.deepcopy(model.state_dict())
                            best_model_weighted = copy.deepcopy(model) 

            scheduler.step()
            print(f'Epoch {epoch} finished')
                
            torch.save(model.state_dict(),
                    Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_e{epoch + 1}.pth'))
    
        torch.save(best_model_samples.state_dict(), Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_bestROCAUC_samples.pth'))
        tb_path.joinpath('detector_bestROCAUC_samples.pth').symlink_to(
            Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_bestROCAUC_samples.pth')
        )

        torch.save(best_model_weighted.state_dict(), Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_bestROCAUC_weighted.pth'))
        tb_path.joinpath('detector_bestROCAUC_weighted.pth').symlink_to(
            Path(__file__).parents[0].joinpath('weights', f'detector_{current_time}_bestROCAUC_weighted.pth')
        )
    
    else: # FLAGS.mode == 'infer'
        
        print('Loading model for inference')
        model.load_state_dict(torch.load(Path(FLAGS.model_path) / 'detector_bestROC.pth'))
        model.eval()
        best_model_samples = model
        best_model_weighted = model

    # # Hack for debug when N_EPOCHS = 0
    # if N_EPOCHS == 0:
    #     best_model = model

    assert (testset.dataset.df.columns == trainset.df.columns).all()
    assert (testset.dataset.df.columns == valset.dataset.df.columns).all()

    best_rocauc_samples_dir = tb_path / 'output_best_rocauc_samples'
    best_rocauc_samples_dir.mkdir()
    best_rocauc_weighted_dir = tb_path / 'output_best_rocauc_weighted'
    best_rocauc_weighted_dir.mkdir()

    u.eval_model(best_model_samples, valloader, testloader, device, best_rocauc_samples_dir)
    u.eval_model(best_model_weighted, valloader, testloader, device, best_rocauc_weighted_dir)

    # GradCAM
    # csv_paths_gradcam = [data_path.joinpath(f'annotation_30_percent_export.csv')]
    # gradcamset = OCTSlicesDataset('test', csv_paths_gradcam, slices_path, TARGET, transform_image=t_test)
    # testloader = torch.utils.data.DataLoader(gradcamset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # pathh = Path(FLAGS.model_path) if FLAGS.model_path else tb_path
    # with open(pathh / 'output_thresholds.json') as json_file:
    #     threshold_json = json.load(json_file)
    # for k, v in threshold_json.items():
    #     threshold_json[k] = float(v)
        
    # u.make_output_images(best_model, testloader, device, tb_path, data_path.joinpath(f'annotation_30_percent_export.csv'), threshold_json)

if __name__ == '__main__':

    from sys import argv

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        default='configs/cfg_detector.yml',
                        type=str,
                        help='Path to the config file')

    parser.add_argument('-m', '--mode',
                        default='train',
                        choices=['train', 'infer'],
                        type=str,
                        help='Modality - train or infer')

    parser.add_argument('--model-path', 
                        required=('infer' in argv), 
                        type=str, 
                        help='Path to the model to be loaded for inference')

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
    TARGET = config['target']
    TRAIN_PERCENT = config['train_percent']
    TEST_PERCENT = config['test_percent']
    writing_per_epoch = config['writing_per_epoch']
    pretrained_model = u.str2bool(config['from_pretrained'])
    try:
        resize = u.str2bool(config['resize'])
    except AttributeError:
        resize = config['resize']
    num_classes = config['classes']

    print(FLAGS.ubelix)
    if FLAGS.ubelix == 0:
        data_path = Path(__file__).parent.joinpath('inputs')
        num_workers = 0
    else:
        data_path = Path('/storage/homefs/ds21n601/oct_biomarker_classification/inputs')
        num_workers = 8

    main()
