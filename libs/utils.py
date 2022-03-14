import torch
import numpy as np
from sklearn import metrics
import argparse
import yaml
import shutil
from collections import OrderedDict


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
    # You'll probably have to change this function
    if type(y_true) == list:
        y_true = np.concatenate(y_true, axis=0)
    if type(y_pred) == list:
        y_pred = np.concatenate(y_pred, axis=0)

    try:
        roc_auc = metrics.roc_auc_score(y_true, y_pred, multi_class='ovr', average='samples')
    except ValueError as e:
        print('Error with ROC AUC', str(e))
        roc_auc = 0

    mAP = metrics.average_precision_score(y_true, y_pred, average='micro')

    prediction_int = np.zeros_like(y_pred)
    prediction_int[y_pred > 0.5] = 1

    mf1 = metrics.f1_score(y_true, prediction_int, average='micro', zero_division=1)
    Mf1 = metrics.f1_score(y_true, prediction_int, average='macro', zero_division=1)

    dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'mF1': mf1, 'MF1': Mf1}
    return dict_metrics
