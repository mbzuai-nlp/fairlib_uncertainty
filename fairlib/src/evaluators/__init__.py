from .evaluator import *
from .utils import *

def present_evaluation_scores(
    valid_preds, valid_labels, valid_private_labels,
    test_preds, test_labels, test_private_labels,
    epoch, epochs_since_improvement, model, epoch_valid_loss,
    is_best, prefix = "checkpoint",
    ):
    """Conduct evaluation, present results, and save evaluation results to file.

    Args:
        valid_preds (np.array): model predictions over the validation dataset.
        valid_labels (np.array): true labels over the validation dataset.
        valid_private_labels (np.array): protected labels over the validation dataset.
        test_preds (np.array): model predictions over the test dataset.
        test_labels (np.array): true labels over the test dataset.
        test_private_labels (np.array): protected labels over the test dataset.
        epoch (float): number of epoch of the model training.
        epochs_since_improvement (int): epoch since the best epoch is updated.
        model (torch.module): the trained model.
        epoch_valid_loss (float): loss over the validation dataset.
        is_best (bool): indicator of whether the current epoch is the best.
        prefix (str, optional): _description_. Defaults to "checkpoint".
    """
    valid_scores, valid_confusion_matrices = gap_eval_scores(
        y_pred=valid_preds,
        y_true=valid_labels, 
        protected_attribute=valid_private_labels,
        args = model.args,
        )
                
    test_scores, test_confusion_matrices = gap_eval_scores(
        y_pred=test_preds,
        y_true=test_labels, 
        protected_attribute=test_private_labels,
        args = model.args,
        )

    # Save checkpoint
    save_checkpoint(
        epoch = epoch, 
        epochs_since_improvement = epochs_since_improvement, 
        model = model, 
        loss = epoch_valid_loss, 
        dev_predictions = valid_preds, 
        test_predictions = test_preds,
        dev_evaluations = valid_scores, 
        test_evaluations = test_scores,
        valid_confusion_matrices = valid_confusion_matrices,
        test_confusion_matrices = test_confusion_matrices,
        is_best = is_best,
        checkpoint_dir = model.args.model_dir,
        prefix = prefix,
        fold=model.args.cross_val_fold)

    validation_results = ["{}: {:2.2f}\t".format(k, 100.*valid_scores[k]) for k in valid_scores.keys()]
    logging.info(('Validation {}').format("".join(validation_results)))
    Test_results = ["{}: {:2.2f}\t".format(k, 100.*test_scores[k]) for k in test_scores.keys()]
    logging.info(('Test {}').format("".join(Test_results)))