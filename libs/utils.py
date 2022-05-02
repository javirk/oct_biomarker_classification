from re import L
import torch
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from scipy.interpolate import interp1d
import argparse
import yaml
import shutil
import json
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def write_to_tb(writer, labels, scalars, iteration, phase='train'):
    for scalar, label in zip(scalars, labels):
        writer.add_scalar(f'{phase}/{label}', scalar, iteration)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
    except shutil.SameFileError:
        pass


def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source, map_location='cpu')
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


def key_transformation(old_key):
    if old_key == 'module._fc.weight':
        return 'module._fc_new.weight'
    if old_key == 'module._fc.bias':
        return 'module._fc_new.bias'

    return old_key


def remove_module_statedict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def calculate_metrics_sigmoid(y_true, y_pred):

    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    try:
        roc_auc_samples = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='samples')
    except ValueError as e:
        print('Error with ROC AUC', str(e))
        roc_auc_samples = 0
    
    try:
        roc_auc_weighted = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')
    except ValueError as e:
        print('Error with ROC AUC', str(e))
        # print(y_true.shape)
        # print(y_true.max(axis=1))
        # print(y_true.max(axis=0))
        roc_auc_weighted = 0

    mAP = metrics.average_precision_score(y_true, y_pred, average='micro')

    # prediction_int = np.zeros_like(y_pred)
    # prediction_int[y_pred > 0.5] = 1 # FIXME: threshold needs to vary

    # mf1 = metrics.f1_score(y_true, prediction_int, average='micro', zero_division=1)
    # Mf1 = metrics.f1_score(y_true, prediction_int, average='macro', zero_division=1)

    dict_metrics = {'ROC AUC samples': roc_auc_samples, 'ROC AUC weighted': roc_auc_weighted, 'mAP': mAP} #, 'mF1': mf1, 'MF1': Mf1}

    return dict_metrics


def run_model_on_dataset(model, data_loader, device):

    model.eval()

    preds, trues, paths = [], [], []

    for data in data_loader:

        inputs, labels, img_paths = data['images'].to(device).float(), data['labels'].to(device), data['paths']

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())
        trues.append(labels.detach().cpu().numpy())
        paths.extend(img_paths)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    return trues, preds, paths


# def create_output_images(model, data_loader, device):
    
#     if isinstance(data_loader.dataset, torch.utils.data.Subset):
#         dataset = data_loader.dataset.dataset

# [ClassifierOutputTarget(category) for category in target_categories]


def eval_model(model, valset, testset, device, save_path):

    if isinstance(testset.dataset, torch.utils.data.Subset):
        label_names = testset.dataset.dataset.df.columns
    else: # torch.utils.data.Dataset
        label_names = testset.dataset.df.columns
    
    cohens_dict = dict()

    ### VALIDATION
    val_trues, val_preds, _ = run_model_on_dataset(model, valset, device)

    # Define thresholds for classes based on optimised sensitivity and specificity on valset
    chosen_thresh = dict() # keep track of best thresholds
    specif, sensit, roc_auc = dict(), dict(), dict()
    cohens_dict['validation'] = dict()

    for i in range(len(label_names)):
        fpr, tpr, thresholds = metrics.roc_curve(val_trues[:, i], val_preds[:, i])
        specif[i], sensit[i] = 1 - fpr, tpr
        roc_auc[i] = metrics.auc(fpr, tpr)

        func = np.sqrt(specif[i]**2 + sensit[i]**2)
        chosen_thresh[i] = thresholds[func.argmax()]

        prediction_int = np.zeros_like(val_preds[:, i])
        prediction_int[val_preds[:, i] > chosen_thresh[i]] = 1

        cohens_dict['validation'][label_names[i]] = str(metrics.cohen_kappa_score(val_trues[:, i], prediction_int))

        fig, ax = plt.subplots()
        fig.suptitle(f'{label_names[i]}\nThreshold: {chosen_thresh[i]:.2f}')
        metrics.ConfusionMatrixDisplay.from_predictions(val_trues[:, i], prediction_int, ax=ax, cmap='Blues')
        fig.tight_layout()
        fig.savefig(f'{save_path}/val_conf_matrix_{"_".join(label_names[i].split())}.png')

    with open(f"{save_path}/output_thresholds.json", "w") as outfile:
        json.dump({label_names[i]: str(chosen_thresh[i]) for i in range(len(label_names))}, outfile)
    
    ### TEST
    test_trues, test_preds, test_paths = run_model_on_dataset(model, testset, device)
    pd.DataFrame(data=test_preds, index=test_paths, columns=label_names).to_csv(save_path/'test_predictions.csv')

    precision, recall = dict(), dict()
    specif, sensit = dict(), dict()
    ap, roc_auc = dict(), dict()
    cohens_dict['test'] = dict()

    # Plot curves
    nrows = 3
    ncols = math.ceil(len(label_names) / nrows)

    fig_ss = plt.figure()
    fig_pr = plt.figure()

    for i in range(len(label_names)):
        precision[i], recall[i], thresholds = metrics.precision_recall_curve(test_trues[:, i], test_preds[:, i])
        ap[i] = metrics.average_precision_score(test_trues[:, i], test_preds[:, i])

        # precision_th = interp1d(thresholds, precision[i][:-1])(chosen_thresh[i])
        # recall_th = interp1d(thresholds, recall[i][:-1])(chosen_thresh[i])

        prediction_int = np.zeros_like(test_preds[:, i])
        prediction_int[test_preds[:, i] > chosen_thresh[i]] = 1

        cohens_dict['test'][label_names[i]] = str(metrics.cohen_kappa_score(test_trues[:, i], prediction_int))

        fig, ax = plt.subplots()
        fig.suptitle(f'{label_names[i]}\nThreshold: {chosen_thresh[i]:.2f}')
        metrics.ConfusionMatrixDisplay.from_predictions(test_trues[:, i], prediction_int, ax=ax, cmap='Blues')
        fig.tight_layout()
        fig.savefig(f'{save_path}/test_conf_matrix_{"_".join(label_names[i].split())}.png')

        axx = fig_pr.add_subplot(nrows, ncols, i + 1)
        axx.plot(recall[i], precision[i])
        # axx.scatter(recall_th, precision_th, c="g", marker=r'$\clubsuit$')
        # axx.plot([0, 1], [0, 1], 'k--')
        axx.set_xlim([0.0, 1.0])
        axx.set_ylim([0.0, 1.05])
        axx.title.set_text('{0}\nAP = {1:0.3f}'.format(label_names[i], ap[i]))
        axx.set_xlabel('Recall')
        axx.set_ylabel('Precision')

        fpr, tpr, thresholds = metrics.roc_curve(test_trues[:, i], test_preds[:, i])
        specif[i], sensit[i] = 1 - fpr, tpr
        roc_auc[i] = metrics.auc(fpr, tpr)

        specif_th = 1 - interp1d(thresholds, fpr)(chosen_thresh[i])
        sensit_th = interp1d(thresholds, tpr)(chosen_thresh[i])

        axx = fig_ss.add_subplot(nrows, ncols, i + 1)
        axx.plot(specif[i], sensit[i])
        axx.scatter(specif_th, sensit_th, c="g", marker=r'$\clubsuit$')
        axx.plot([0, 1], [1, 0], 'k--')
        axx.set_xlim([0.0, 1.0])
        axx.set_ylim([0.0, 1.05])
        axx.title.set_text('{0}\nAUC = {1:0.3f}'.format(label_names[i], roc_auc[i]))
        axx.set_xlabel('Specificity')
        axx.set_ylabel('Sensitivity')

    with open(f"{save_path}/output_kappa_score.json", "w") as outfile:
        json.dump(cohens_dict, outfile)

    fig_pr.tight_layout()
    fig_pr.savefig(f'{save_path}/pr_curve.png')

    fig_ss.tight_layout()
    fig_ss.savefig(f'{save_path}/roc_curve.png')

    return thresholds


def make_output_images(model, dataloader, device, save_path, ref_file, threshold_dict):
    
    ref_df = pd.read_csv(ref_file)

    img_save_path = save_path / 'gradcam'
    img_save_path.mkdir()

    model.eval()

    target_layers = [model.module.layer4[-1]]
    labels = dataloader.dataset.dataset.df.columns

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[3, 4])

    ax_img = fig.add_subplot(gs[:2, 0])

    gs_gradcam = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs[:2, 1], hspace=0.1, wspace=0.1)
    axs_gradcam = []
    for i in range(len(labels)):
        axs_gradcam.append(fig.add_subplot(gs_gradcam[i // 4, i % 4]))

    gs_tables = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[-1, :])
    ax_table_ref = fig.add_subplot(gs_tables[:6])
    ax_table_maj = fig.add_subplot(gs_tables[6])
    ax_table_pred = fig.add_subplot(gs_tables[7])

    for batch in dataloader:

        inputs, img_paths = batch['images'].to(device).float(), batch['paths']

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        outputs = outputs.sigmoid().detach().cpu().numpy()
        
        # Iterate through batch
        for input_img, img_path, pred in zip(inputs, img_paths, outputs):
         
            norm_img = input_img.cpu().numpy()[0] # first channel is grey, dataloader stacks them for resnet
            scaled_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
            bgr_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)

            for lbl_idx in range(len(labels)):

                targets = [ClassifierOutputTarget(lbl_idx)]    
                grayscale_cam = cam(input_tensor=torch.unsqueeze(input_img, 0),
                                    targets=targets, 
                                    aug_smooth=True, 
                                    eigen_smooth=False)
                gradcam_img = show_cam_on_image(bgr_img, grayscale_cam[0, :], use_rgb=False, colormap=cv2.COLORMAP_RAINBOW)

                ax_img.imshow(bgr_img)
                ax_img.axis('off')

                axs_gradcam[lbl_idx].imshow(gradcam_img, alpha=1.0 if pred[lbl_idx] > threshold_dict[labels[lbl_idx]] else 0.25)
                axs_gradcam[lbl_idx].axis('off')
                # axs_gradcam[lbl_idx].set_title(labels[lbl_idx], fontdict={'fontsize': 6})
                axs_gradcam[lbl_idx].text(0.02, 0.02, labels[lbl_idx],
                                          color='white', fontsize='x-small',
                                          horizontalalignment='left', 
                                          verticalalignment='bottom', 
                                          transform=axs_gradcam[lbl_idx].transAxes)

            image_df = ref_df[ref_df.filename == img_path][['biomarker', 'grader1', 'grader2', 'grader3', 'grader4', 'grader5', 'majority']]
            entr_df = image_df[~image_df.filter(like='grader').apply(set, axis=1).isin([{False, np.nan}, {False}])].set_index('biomarker')
            image_df = image_df.set_index('biomarker').reindex(labels)

            entropy = round((entr_df.filter(like='grader') == False).sum().sum() /
            ((entr_df.filter(like='grader') == False).sum().sum() + entr_df.filter(like='grader').sum().sum() ) * 100)

            ax_table_ref.axis('tight')
            ax_table_ref.axis('off')
            colors = image_df.applymap(lambda x: '#9BCA3E' if x == True else ('#ED5314' if x == False else '#C5C7D8'))
            # print(colors.values.shape)
            tabb = ax_table_ref.table(
                cellText=image_df.filter(like='grader').values,
                colLabels=image_df.filter(like='grader').columns,
                rowLabels=image_df.index,
                loc='center right',
                cellColours=colors.filter(like='grader').values,
                # bbox=[0.4, 0, 0.6, 1.0]
                )
            tabb.scale(0.8, 1)

            ax_table_maj.axis('tight')
            ax_table_maj.axis('off')
            ax_table_maj.table(
                cellText=image_df[['majority']].values,
                colLabels=['majority'],
                loc='center',
                cellColours=colors[['majority']].values
                )

            pred_df = pd.DataFrame(pred, index=labels, columns=['AI'])
            colors = pred_df.apply(lambda row: '#9BCA3E' if row['AI'] > threshold_dict[row.name] else '#ED5314', axis=1)
            ax_table_pred.axis('tight')
            ax_table_pred.axis('off')
            ax_table_pred.table(
                cellText=np.round(pred_df[['AI']].values, 3),
                colLabels=['AI'],
                loc='center',
                cellColours=np.expand_dims(colors.values, axis=1)
                )

            fig.tight_layout()
            fig.savefig(img_save_path / (str(entropy).zfill(2) + '_' + img_path))

            ax_img.clear()
            ax_table_ref.clear()
            ax_table_maj.clear()
            ax_table_pred.clear()
            for ax in axs_gradcam:
                ax.clear()

            plt.close(fig)


def compute_contrast(image_dir: pathlib.Path) -> None:

    img_paths = image_dir.glob('*.png')
    contrast_list = []

    for i, img_path in enumerate(img_paths):

        # print(img_path)

        img = cv2.imread(str(img_path))
        # Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

        # compute min and max of Y
        min = np.min(img)
        max = np.max(img)

        # compute contrast
        contrast = (max-min)/(max+min)
        contrast_list.append(contrast)

        if (i + 1) % 100 == 0:
            print(i + 1)

    plt.hist(contrast_list, 20)
    plt.savefig('hihi.png')


if __name__ == '__main__':

    img_dir = pathlib.Path(__file__).parent.parent.absolute() / 'inputs' / 'slices'
    compute_contrast(img_dir)

