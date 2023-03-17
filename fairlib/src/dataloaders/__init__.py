import sys
import torch
import numpy as np
import logging
from .loaders import default_dataset_roots, name2loader
from . import utils
from .encoder import text2id
from collections import defaultdict

def get_dataloaders(args):
    """Initialize the torch dataloaders according to arguments.

    Args:
        args (namespace): arguments

    Raises:
        NotImplementedError: if correspoding components have not been implemented.

    Returns:
        tuple: dataloaders for training set, development set, and test set.
    """
    task_dataloader = name2loader(args)
    
    if args.encoder_architecture in ["Fixed", "MNIST"]:
        pass
    elif args.encoder_architecture == "BERT":
        # Init the encoder form text to idx.
        args.text_encoder = text2id(args)
    else:
        raise NotImplementedError

    train_data = task_dataloader(args=args, split="train")
    dev_data = task_dataloader(args=args, split="dev")
    test_data = task_dataloader(args=args, split="test")
    if args.remove_percent != 0:
        if args.num_clusters != 0:
            train_data.cluster_and_remove(test_data)
        else:
            train_data.remove_similar(test_data)
        
    if args.cross_val and (args.cross_val_fold is not None):
        X = np.concatenate([train_data.X, dev_data.X])
        y = np.concatenate([train_data.y, dev_data.y])
        protected_label = np.concatenate([train_data.protected_label, dev_data.protected_label])

        if args.encoder_architecture == "BERT":
            token_type_ids = np.concatenate([train_data.token_type_ids, dev_data.token_type_ids])
            mask = np.concatenate([train_data.mask, dev_data.mask])

        train_data.X, dev_data.X = X[args.train_idx], X[args.dev_idx]
        train_data.y, dev_data.y = y[args.train_idx], y[args.dev_idx]
        train_data.protected_label, dev_data.protected_label = protected_label[args.train_idx], protected_label[args.dev_idx]

        if args.encoder_architecture=="BERT":
            train_data.token_type_ids, dev_data.token_type_ids = token_type_ids[args.train_idx], token_type_ids[args.dev_idx]
            train_data.mask, dev_data.mask = mask[args.train_idx], mask[args.dev_idx]


    # DataLoader Parameters
    train_dataloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers}

    eval_dataloader_params = {
            'batch_size': args.test_batch_size,
            'shuffle': False,
            'num_workers': args.num_workers}

    # init dataloader
    training_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
    validation_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
    test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

    return training_generator, validation_generator, test_generator