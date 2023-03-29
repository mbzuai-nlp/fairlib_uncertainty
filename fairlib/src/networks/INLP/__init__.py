from .debias import get_debiasing_projection, get_projection_to_intersection_of_nullspaces

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from collections import Counter, defaultdict

import warnings
warnings.filterwarnings('ignore')

import torch
from pathlib import Path
import numpy as np

from ...evaluators import present_evaluation_scores
from ...evaluators.evaluator import gap_eval_scores
import logging

def load_trained_model(model, checkpoint_dir, device):
    checkpoint_PATH = Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar'
    model.load_state_dict(torch.load(checkpoint_PATH)["model"])
    model.to(device)
    model.eval()
    return model

def get_INLP_trade_offs(model, args):
    # Hyperparameters
    discriminator_reweighting = args.INLP_discriminator_reweighting
    by_class = args.INLP_by_class
    n = args.INLP_n
    min_acc = args.INLP_min_acc
    is_autoregressive = True
    dim = args.hidden_size
    clf = LogisticRegression
    clf_params = {'fit_intercept': True, 'class_weight': discriminator_reweighting, 'dual': False, 'C': 0.1, "max_iter": 100}

    # Load best checkpoints
    model = load_trained_model(model, args.model_dir, args.device)

    # Extract Hidden representations
    train_hidden, train_labels, train_private_labels, train_regression_labels = model.extract_hidden_representations("train")
    dev_hidden, dev_labels, dev_private_labels, dev_regression_labels = model.extract_hidden_representations("dev")
    test_hidden, test_labels, test_private_labels, test_regression_labels = model.extract_hidden_representations("test")

    # Run INLP

    P_n = get_debiasing_projection(clf, clf_params, n, dim, is_autoregressive, min_acc,
                                    train_hidden, train_private_labels, dev_hidden, dev_private_labels,
                                    by_class=by_class, Y_train_main=train_labels, Y_dev_main=dev_labels)
    
    rowspaces = P_n[1]
    best_DTO = 100.0
    valid_accs = []
    valid_fairs = []
    best_valid_dto = 1e+5
    best_epoch = 0

    for iteration, p_iteration in enumerate(range(1, len(rowspaces))):
        
        P = get_projection_to_intersection_of_nullspaces(rowspaces[:p_iteration], input_dim=train_hidden.shape[1])
        
        debiased_x_train = P.dot(train_hidden.T).T
        debiased_x_dev = P.dot(dev_hidden.T).T
        debiased_x_test = P.dot(test_hidden.T).T
        
        if not args.regression:
            classifier = LogisticRegression(warm_start = True, 
                                                penalty = 'l2',
                                                solver = "sag", 
                                                multi_class = 'multinomial', 
                                                fit_intercept = True,
                                                verbose = 0, 
                                                max_iter = 10,
                                                n_jobs = 24, 
                                                random_state = 1)
            classifier.fit(debiased_x_train, train_labels)
        else:
            classifier = LinearRegression()
            classifier.fit(debiased_x_train, train_regression_labels)

        
        
        # Evaluation
        dev_y_pred = classifier.predict(debiased_x_dev)
        test_y_pred= classifier.predict(debiased_x_test)

        logging.info("Evaluation at Epoch %d" % (iteration,))

        # check if model is best by DTO on val set
        valid_scores, valid_confusion_matrices = gap_eval_scores(
            y_pred=dev_y_pred,
            y_true=dev_labels if not args.regression else dev_regression_labels, 
            protected_attribute=dev_private_labels,
            args = model.args,
        )
        is_best = False
        curr_DTO = np.sqrt((1 - valid_scores["accuracy"]) ** 2 + (valid_scores["TPR_GAP"]) ** 2)
        if curr_DTO < best_DTO:
            best_DTO = curr_DTO
            is_best = True

        present_evaluation_scores(
            valid_preds = dev_y_pred, 
            valid_labels = dev_labels if not args.regression else dev_regression_labels, 
            valid_private_labels = dev_private_labels,
            test_preds = test_y_pred, 
            test_labels = test_labels if not args.regression else test_regression_labels, 
            test_private_labels = test_private_labels,
            epoch = iteration, epochs_since_improvement = None, 
            model = model, epoch_valid_loss = None,
            is_best = is_best, prefix = "INLP_checkpoint",
            )
        _state = {
            'classifier': classifier,
            'P': P}

        # also find best checkpoint by bdto
        valid_accs.append(valid_scores["accuracy"])
        valid_fairs.append((1 - valid_scores["TPR_GAP"]))
        # now calc balanced dto - normalize accuracy and fairness on [0, 1] on all epochs
        if len(valid_accs) > 1:
            if args.early_stopping_criterion == "balanced_dto":
                balanced_fairs = (np.array(valid_fairs) - np.min(valid_fairs)) / (np.max(valid_fairs) - np.min(valid_fairs))
                balanced_accs = (np.array(valid_accs) - np.min(valid_accs)) / (np.max(valid_accs) - np.min(valid_accs))
            else:
                # bdto with max norm
                balanced_fairs = np.array(valid_fairs) / np.max(valid_fairs)
                balanced_accs = np.array(valid_accs) / np.max(valid_accs)
        else:
            balanced_fairs = np.array(valid_fairs)
            balanced_accs = np.array(valid_accs)
        if any(np.array(args.early_stopping_weights) != 1.0):
            weights = args.early_stopping_weights
            logging.info("Use WBDTO with weights: " + str(args.early_stopping_weights))
            epoch_valid_dto = np.sqrt(weights[0] * (1 - balanced_accs[-1]) ** 2 + weights[1] * (1 - balanced_fairs[-1]) ** 2)
        else:
            epoch_valid_dto = np.sqrt((1 - balanced_accs[-1]) ** 2 + (1 - balanced_fairs[-1]) ** 2)
        # we also have to renormalize best_valid_dto each time
        if iteration > 0:
            if any(np.array(args.early_stopping_weights) != 1.0):
                weights = args.early_stopping_weights
                best_valid_dto = np.sqrt(weights[0] * (1 - balanced_accs[best_epoch]) ** 2 + weights[1] * (1 - balanced_fairs[best_epoch]) ** 2)
            else:
                best_valid_dto = np.sqrt((1 - balanced_accs[best_epoch]) ** 2 + (1 - balanced_fairs[best_epoch]) ** 2)
        is_best_bdto = epoch_valid_dto < best_valid_dto
        best_epoch = iteration if is_best else best_epoch
        best_valid_dto = min(epoch_valid_dto, best_valid_dto)
        if is_best_bdto:
            # save both model and classifier
            present_evaluation_scores(
                valid_preds = dev_y_pred, 
                valid_labels = dev_labels if not args.regression else dev_regression_labels, 
                valid_private_labels = dev_private_labels,
                test_preds = test_y_pred, 
                test_labels = test_labels if not args.regression else test_regression_labels, 
                test_private_labels = test_private_labels,
                epoch = iteration, epochs_since_improvement = None, 
                model = model, epoch_valid_loss = None,
                is_best = is_best_bdto, prefix = "INLP_bdto_checkpoint",
                )
            filename = "INLP_bdto_checkpoint_epoch_cls_BEST" + '.pth.tar'
            torch.save(_state, Path(model.args.model_dir) / filename)
        if is_best:
            filename = "INLP_checkpoint_epoch_cls_BEST" + '.pth.tar'
            torch.save(_state, Path(model.args.model_dir) / filename)

        filename = "INLP_checkpoint_epoch_cls{:.2f}".format(iteration) + '.pth.tar'
        torch.save(_state, Path(model.args.model_dir) / filename)

    return None
