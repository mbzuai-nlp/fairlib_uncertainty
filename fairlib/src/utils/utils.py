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
    X = np.concatenate([state.opt.train_generator.dataset.X, state.opt.dev_generator.dataset.X])
    y = np.concatenate([state.opt.train_generator.dataset.y, state.opt.dev_generator.dataset.y])
    protected_label = np.concatenate([state.opt.train_generator.dataset.protected_label, state.opt.dev_generator.dataset.protected_label])

    if state.encoder_architecture=="BERT":
        token_type_ids = np.concatenate([state.opt.train_generator.dataset.token_type_ids, state.opt.dev_generator.dataset.token_type_ids])
        mask = np.concatenate([state.opt.train_generator.dataset.mask, state.opt.dev_generator.dataset.mask])
        
    kf = KFold(n_splits=5, random_state=state.base_seed, shuffle=True)
    for i, (train_index, dev_index) in enumerate(kf.split(X)):
        state.opt.train_generator.dataset.X, state.opt.dev_generator.dataset.X = X[train_index], X[dev_index]
        state.opt.train_generator.dataset.y, state.opt.dev_generator.dataset.y = y[train_index], y[dev_index]
        state.opt.train_generator.dataset.protected_label, state.opt.dev_generator.dataset.protected_label = protected_label[train_index], protected_label[dev_index]

        if state.encoder_architecture=="BERT":
            state.opt.train_generator.dataset.token_type_ids, state.opt.dev_generator.dataset.token_type_ids = token_type_ids[train_index], token_type_ids[dev_index]
            state.opt.train_generator.dataset.mask, state.opt.dev_generator.dataset.mask = mask[train_index], mask[dev_index]

        yield i, state
