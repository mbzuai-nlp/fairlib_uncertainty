import numpy as np
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel
from transformers import (
    ElectraForSequenceClassification,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
#from timm.loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from torch.autograd import Variable


def entropy(x):
    return torch.sum(-x * torch.log(torch.clamp(x, 1e-8, 1)), axis=-1)

def conf(preds, probs, labels):
    conf_scores = torch.where(preds == labels, torch.max(probs, axis=-1).values, 1 - torch.max(probs, axis=-1).values)
    return conf_scores

class AsymmetricCELoss(nn.Module):
    def __init__(self, margin=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricCELoss, self).__init__()
        
        self.margin = margin
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.margin is not None and self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()
    
def multilabel_loss(probs, labels, margin=0.05, gamma_neg=4, gamma_pos=1):
    loss_func = AsymmetricLossMultiLabel(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=margin)
    labels_ohe = torch.nn.functional.one_hot(labels, num_classes=probs.shape[-1])
    loss = loss_func(probs, labels_ohe)

    return loss 
    
def RAU_loss(probs, labels, unc_threshold=0.5, eps=1e-6):
    preds = torch.argmax(probs, axis=-1)
    conf_scores = conf(preds, probs, labels)
    uncertainty = entropy(probs)
    n_C = conf_scores * (1 - torch.tan(uncertainty))
    n_U = conf_scores * (torch.tan(uncertainty))
    
    n_AC = torch.where((preds == labels) & (uncertainty <= unc_threshold), n_C, torch.tensor(0.).to(labels.device)).sum()
    n_AU = torch.where((preds == labels) & (uncertainty > unc_threshold), n_U, torch.tensor(0.).to(labels.device)).sum()
    n_IC = torch.where((preds != labels) & (uncertainty <= unc_threshold), n_C, torch.tensor(0.).to(labels.device)).sum()
    n_IU = torch.where((preds != labels) & (uncertainty > unc_threshold), n_U, torch.tensor(0.).to(labels.device)).sum()
    loss = torch.log(1 + n_AU / (n_AC + n_AU + eps) + n_IC / (n_IC + n_IU + eps))
    return loss 

def multiclass_metric_loss_fast(represent, target, margin=10, class_num=2, start_idx=1,
                                per_class_norm=False):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0
    for i in range(class_num):
        curr_repr = represent[indices[i]]
        s_k = len(indices[i])
        triangle_matrix = torch.triu(
            (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
        )
        buf_loss = torch.sum(1 / dim * (triangle_matrix ** 2))
        if per_class_norm:
            loss_intra += buf_loss / np.max([(s_k ** 2 - s_k), 1]) * 2
        else:
            loss_intra += buf_loss
            num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2
        for j in range(i + 1, class_num):
            repr_j = represent[indices[j]]
            s_q = len(indices[j])
            matrix = (curr_repr.unsqueeze(1) - repr_j).norm(2, dim=-1)
            inter_buf_loss = torch.sum(torch.clamp(margin - 1 / dim * (matrix ** 2), min=0))
            if per_class_norm:
                loss_inter += inter_buf_loss / np.max([(s_k * s_q), 1])
            else:
                loss_inter += inter_buf_loss
                num_inter += repr_j.shape[0] * curr_repr.shape[0]
    if num_intra > 0 and not(per_class_norm):
        loss_intra = loss_intra / num_intra
    if num_inter > 0 and not(per_class_norm):
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter


def compute_loss_cer(logits, labels, loss, lamb, unpad=False):
    """Computes regularization term for loss with CER
    """
    # here correctness is always 0 or 1
    if unpad:
        # NER case
        logits = logits[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    # suppose that -1 will works for ner and cls
    confidence, prediction = torch.softmax(logits, dim=-1).max(dim=-1)
    correctness = prediction == labels
    correct_confidence = torch.masked_select(confidence, correctness)
    wrong_confidence = torch.masked_select(confidence, ~correctness)
    regularizer = 0
    regularizer = torch.sum(
        torch.clamp(wrong_confidence.unsqueeze(1) - correct_confidence, min=0)
        ** 2
    )
    loss += lamb * regularizer
    return loss


def compute_loss_metric(hiddens, labels, loss, num_labels,
                        margin, lamb_intra, lamb, unpad=False):
    """Computes regularization term for loss with Metric loss
    """
    if unpad:
        hiddens = hiddens[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    class_num = num_labels
    start_idx = 0 if class_num == 2 else 1
    # TODO: define represent, target and margin
    # Get only sentence representaions
    loss_intra, loss_inter = multiclass_metric_loss_fast(
        hiddens,
        labels,
        margin=margin,
        class_num=class_num,
        start_idx=start_idx,
    )
    loss_metric = lamb_intra * loss_intra[0] + lamb * loss_inter[0]
    loss += loss_metric
    return loss


class AsymmetricLossSingleLabel(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(AsymmetricLossSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (1-hot vector)
        """

        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.nn.functional.one_hot(target, num_classes=inputs.shape[-1])#torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
            
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
