import random
import os
import difflib
import numpy as np
from sklearn.model_selection import KFold
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def diff_str(first, second):
    firstlines = first.splitlines(keepends=True)
    secondlines = second.splitlines(keepends=True)
    if len(firstlines) == 1 and first.strip('\r\n') == first:
        firstlines = [first + '\n']
        secondlines = [second + '\n']
    return ''.join(difflib.unified_diff(firstlines, secondlines))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        
        
def kfold_train_dev(state):
    X = np.arange(len(state.opt.train_generator.dataset.X) + len(state.opt.dev_generator.dataset.X))
    kf = KFold(n_splits=state.cross_val_n_splits, random_state=state.base_seed, shuffle=True)
    for i, (train_index, dev_index) in enumerate(kf.split(X)):
        yield i, (train_index, dev_index)
