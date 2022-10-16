from sklearn import ensemble
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
from random import randint
from time import sleep
import os
import copy
import json
from libs import resnet

RNDM = 91
torch.manual_seed(RNDM)
DIR_UBELIX = Path(__file__).parent.joinpath("inputs")
DIR_LOCAL = Path("/storage/homefs/ds21n601/oct_biomarker_classification/inputs")
DIR_SLICES = "slices"

class EnsembleModel(nn.Module):

    def __init__(self, model_paths, model_name, nb_classes, device):
        super(EnsembleModel, self).__init__()

        self.models = []
        
        for model_path in model_paths:
            
            model = getattr(resnet, model_name)(pretrained=False, num_classes=nb_classes) # we don't need pretrained as we load it
            # model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            state = torch.load(Path(model_path))
            state = {k.replace('module.', ''): v for k, v in state.items()} # it was enclosed in nn.DataParallel
            model.load_state_dict(state)

            for param in model.parameters():
                param.requires_grad_(False)
            
            model.to(device) # FIXME: so far it works only with 1 GPU
            self.models.append(model)

        # # Remove last linear layer
        # self.modelA.fc = nn.Identity()
        # self.modelB.fc = nn.Identity()
    
        # Create new classifier
        self.classifier = nn.Linear(len(self.models)*nb_classes, nb_classes)
    
    def forward(self, x):

        preds = []
        for model in self.models:
            pred = model(x.clone())
            # pred = pred.sigmoid() # FIXME: remove?
            pred = pred.view(pred.size(0), -1)
            preds.append(pred) # clone to make sure x is not changed by inplace methods

        out = torch.cat(preds, dim=1)
        out = self.classifier(out)
        return out


class OCTMultiLabelDetector:

    def __init__(self, args) -> None:
        
        self.data_path = DIR_UBELIX if args.ubelix else DIR_LOCAL
        self.slices_path = self.data_path.joinpath(DIR_SLICES)

        self.current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if args.command == 'train':
            os.makedirs('weights', exist_ok=True)
            path_str = "DET_{}_{}_train{:03d}_test{:03d}_ep{:02d}_bs{:03d}_lr{:.2E}_train-{}_test-{}".format(
                args.command.upper(),
                self.current_time,
                args.train_data,
                args.test_data,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.train_target,
                args.test_target
            )
        else:
            path_str = "DET_{}_{}_test{:03d}_test-{}".format(
                args.command.upper(),
                self.current_time,
                args.test_data,
                args.test_target
            )

        if getattr(args, 'restart_from', None) is not None:
            path_str += "_RESTART"

        if getattr(args, 'ensemble', False):
            path_str += "_ENSEMBLE"

        # path_str += "_VERTSHIFT"

        self.tb_path = Path(__file__).resolve().parents[0].joinpath("runs", path_str)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        self.writer = SummaryWriter(self.tb_path)

        with open(self.tb_path / 'commandline_args.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        self.args = args
        # TODO: implement cerberus validation on args

    def load_datasets(self):

        if self.args.resize:
            t = transforms.Compose([
                Resize(self.args.resize), 
                transforms.ToTensor(), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(10),
                # transforms.RandomAffine(degrees=0, translate=(0, 0.1)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]) 
            t_test = transforms.Compose([
                Resize(self.args.resize), 
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]) 
        else:
            t = transforms.Compose([
                transforms.ToTensor(), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(10),
                # transforms.RandomAffine(degrees=0, translate=(0, 0.1)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            t_test = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]) 

        if self.args.command == 'train':
            csv_paths_train = [self.data_path.joinpath(f'annotation_{perc:03d}_percent_export.csv') for perc in range(10, self.args.train_data + 1, 10)]    
            trainset = OCTSlicesDataset('train', csv_paths_train, self.slices_path, self.args.train_target, transform_image=t)
            self.trainset_size = len(trainset)
            self.writing_freq_train = self.trainset_size // (self.args.writing_per_epoch * self.args.batch_size)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        else:
            trainset = None
            self.trainset_size = -1
            self.writing_freq_train = None
            self.trainloader = None

        csv_paths_test = [self.data_path.joinpath(f'annotation_{perc:03d}_percent_export.csv') for perc in range(10, self.args.test_data + 1, 10)]
        testvalset = OCTSlicesDataset('test', csv_paths_test, self.slices_path, target=self.args.test_target, transform_image=t_test, split_on='majority')

        lbls_encoded = LabelEncoder().fit_transform([''.join(str(l)) for l in testvalset.label_split])
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=RNDM)
        valset_indices, testset_indices = list(sgkf.split(testvalset.image_set, lbls_encoded, groups=testvalset.patient_set))[0] # it was volume_set
        valset_size = len(valset_indices)
        testset_size = len(testset_indices)

        with open(self.tb_path / 'validation_indices.csv', 'w') as fp:
            fp.write('\n'.join([str(ii) for ii in valset_indices]))

        with open(self.tb_path / 'test_indices.csv', 'w') as fp:
            fp.write('\n'.join([str(ii) for ii in testset_indices]))

        valset = torch.utils.data.Subset(testvalset, valset_indices)
        testset = torch.utils.data.Subset(testvalset, testset_indices)

        self.writing_freq_val = valset_size // self.args.batch_size  # Only once per epoch
        self.writing_freq_test = testset_size // self.args.batch_size  # Only once per epoch

        self.valloader = torch.utils.data.DataLoader(valset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        print(self.trainset_size, valset_size, testset_size)

        pd.DataFrame(
            zip(testvalset.df.iloc[testset_indices].sum(), testvalset.df.iloc[valset_indices].sum()), 
            columns=['test', 'val'], 
            index=testvalset.df.columns
        ).to_csv(self.tb_path / 'val_test_distribution.csv')

    def load_model(self):

        if getattr(self.args, 'ensemble', False):
            model = EnsembleModel(self.args.model_paths, self.args.model_name, self.args.classes, self.device)
        else:
            model = getattr(resnet, self.args.model_name)(pretrained=getattr(self.args, 'pretrained', False), num_classes=self.args.classes)
        
        print(f'GPU devices: {torch.cuda.device_count()}')
        self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(self.device)

        if getattr(self.args, 'restart_from', None) is not None:
            self.model.load_state_dict(torch.load(Path(self.args.restart_from)))

        if self.args.command == 'infer':
            print('Loading model for inference')
            self.model.load_state_dict(torch.load(Path(self.args.model_path)))
            self.model.eval()

    def train(self):
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6, momentum=0.9)
        lmbda = lambda epoch: 0.99
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

        print(f'Applied weights for training loss will be: {self.trainloader.dataset.posweights.numpy()}')

        criterion_train = nn.BCEWithLogitsLoss(pos_weight=self.trainloader.dataset.posweights.to(self.device)) 
        criterion_val   = nn.BCEWithLogitsLoss(pos_weight=self.valloader.dataset.dataset.posweights.to(self.device)) 
        criterion_test  = nn.BCEWithLogitsLoss(pos_weight=self.testloader.dataset.dataset.posweights.to(self.device)) 

        best_rocauc_samples = 0.0
        best_rocauc_weighted = 0.0
        
        for epoch in range(self.args.epochs):

            for phase in ['train', 'test', 'validation']:
                running_loss = 0.0
                running_pred = []
                running_true = []

                if phase == 'train':
                    self.model.train()
                    loader = self.trainloader
                    writing_freq = self.writing_freq_train
                    criterion = criterion_train
                    i_train = 0
                elif phase == 'validation':
                    self.model.eval()
                    loader = self.valloader
                    writing_freq = self.writing_freq_val
                    criterion = criterion_val
                elif phase == 'test':
                    self.model.eval()
                    loader = self.testloader
                    writing_freq = self.writing_freq_test
                    criterion = criterion_test

                for i, data in enumerate(loader):
                    inputs, labels = data['images'].to(self.device).float(), data['labels'].to(self.device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
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

                        n_epoch = epoch * self.trainset_size // self.args.batch_size + i_train + 1
                        epoch_loss = running_loss / (writing_freq * self.args.batch_size)
                        dict_metrics = u.calculate_metrics_sigmoid(running_true, running_pred)
                        epoch_rocauc_samples = dict_metrics['ROC AUC samples']
                        epoch_rocauc_weighted = dict_metrics['ROC AUC weighted']
                        print(f'{phase} Loss: {epoch_loss} ROC AUC: {epoch_rocauc_samples}')
                        dict_metrics['Loss'] = epoch_loss
                        u.write_to_tb(self.writer, dict_metrics.keys(), dict_metrics.values(), n_epoch, phase=phase)

                        running_pred = []
                        running_true = []
                        running_loss = 0.0

                        if phase == 'validation' and epoch_rocauc_samples > best_rocauc_samples: 
                            best_rocauc_samples = epoch_rocauc_samples
                            # best_model_wts = copy.deepcopy(model.state_dict())
                            best_model_samples = copy.deepcopy(self.model) 

                        if phase == 'validation' and epoch_rocauc_weighted > best_rocauc_weighted: 
                            best_rocauc_weighted = epoch_rocauc_weighted
                            # best_model_wts = copy.deepcopy(model.state_dict())
                            best_model_weighted = copy.deepcopy(self.model) 

            scheduler.step()
            print(f'Epoch {epoch + 1} finished')
                
            # torch.save(self.model.state_dict(),
            #         Path(__file__).parents[0].joinpath('weights', f'detector_{self.current_time}_e{epoch + 1}.pth'))
    
        # Save best models and create symlink in working directories
        best_rocauc_model_path = Path(__file__).parents[0].joinpath(
            'weights', f'detector_{self.current_time}_bestROCAUC_samples.pth'
        )
        torch.save(best_model_samples.state_dict(), best_rocauc_model_path)
        (self.tb_path / 'output_best_rocauc_samples').mkdir()
        self.tb_path.joinpath('output_best_rocauc_samples', 'detector_bestROCAUC_samples.pth').symlink_to(best_rocauc_model_path)

        best_rocauc_model_path = Path(__file__).parents[0].joinpath(
            'weights', f'detector_{self.current_time}_bestROCAUC_weighted.pth'
        )
        torch.save(best_model_weighted.state_dict(), best_rocauc_model_path)
        (self.tb_path / 'output_best_rocauc_weighted').mkdir()
        self.tb_path.joinpath('output_best_rocauc_weighted', 'detector_bestROCAUC_weighted.pth').symlink_to(best_rocauc_model_path)

        self.infer(best_model_samples, self.tb_path / 'output_best_rocauc_samples')
        self.infer(best_model_weighted, self.tb_path / 'output_best_rocauc_weighted')

    # def compute_thresholds():
    #     pass

    def load_thresholds():
        pass

    def infer(self, model, save_dir, gradcam=False):

        save_dir.mkdir(exist_ok=True)

        # assert (self.testloader.dataset.dataset.df.columns == self.trainloader.dataset.df.columns).all()
        assert (self.testloader.dataset.dataset.df.columns == self.valloader.dataset.dataset.df.columns).all()

        u.eval_model(model, self.valloader, self.testloader, self.device, save_dir)

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

