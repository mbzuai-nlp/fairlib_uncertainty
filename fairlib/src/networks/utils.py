import torch.nn as nn
import torch
import logging
from torch.optim import Adam
import time
from pathlib import Path
from ..evaluators import print_network, present_evaluation_scores
from ..evaluators.evaluator import gap_eval_scores
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# train the main model with adv loss
def train_epoch(model, iterator, args, epoch):

    epoch_loss = 0
    model.train()

    optimizer = model.optimizer
    criterion = model.criterion
    
    data_t0 = time.time()
    data_t, t = 0, 0
    
    for it, batch in enumerate(iterator):
        text = batch[0]
        if len(text.shape) > 2:
            text = text.squeeze()
        if len(batch) == 7:
            if not torch.is_tensor(batch[6]):
                mask = torch.stack(batch[6])
            else:
                mask = batch[6]
            mask = mask.float().to(args.device)
            if len(mask.shape) > 1:
                mask = mask.squeeze()

        tags = batch[1].long()
        if len(tags.shape) > 1:
            tags = tags.squeeze()
        p_tags = batch[2].float()
        if len(p_tags.shape) > 1:
            p_tags = p_tags.squeeze()

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(args.device)
        
        if args.regression:
            regression_tags = batch[5].float()
            if len(regression_tags.shape) > 1:
                regression_tags = regression_tags.squeeze()
            regression_tags = regression_tags.to(args.device)

        text = text.to(args.device)
        tags = tags.to(args.device)
        p_tags = p_tags.to(args.device)

        data_t += (time.time() - data_t0)
        t0 = time.time()
        
        optimizer.zero_grad()
        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            if len(batch) == 7:
                predictions = model(text, mask)
            else:
                predictions = model(text)

        if len(predictions.shape) > 1:
            predictions = predictions if not args.regression else predictions.squeeze()
        # main tasks loss
        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags if not args.regression else regression_tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags if not args.regression else regression_tags)

        if args.adv_debiasing:
            # Update discriminator if needed
            if args.adv_update_frequency == "Batch":
                args.discriminator.train_self_batch(model, batch)

            # get hidden representations
            if args.gated:
                if len(batch) == 7:
                    hs = model.hidden(text, mask, p_tags)
                else:
                    hs = model.hidden(text, p_tags)
            else:
                if len(batch) == 7:
                    hs = model.hidden(text, mask)
                else:
                    hs = model.hidden(text)

            adv_losses = args.discriminator.adv_loss(hs, tags, p_tags)

            for adv_loss in adv_losses:
                loss = loss - (adv_loss / args.adv_num_subDiscriminator)

        if args.FCL:
            # get hidden representations
            if args.gated:
                if len(batch) == 7:
                    hs = model.hidden(text, mask, p_tags)
                else:
                    hs = model.hidden(text, p_tags)
            else:
                if len(batch) == 7:
                    hs = model.hidden(text, mask)
                else:
                    hs = model.hidden(text)

            # update the loss with Fair Supervised Contrastive Loss
            fscl_loss = args.FairSCL(hs, tags, p_tags)
            loss = loss + fscl_loss
        
        if (args.DyBT is not None) and (args.DyBT == "GroupDifference"):
            loss = loss + args.group_difference_loss(
                predictions, tags, p_tags, 
                regression_tags = None if not args.regression else regression_tags,
                )

        loss.backward()

        # Zero gradients of the cls head 
        if it % args.classification_head_update_frequency != 0:
            model.zero_cls_grad()

        optimizer.step()
        epoch_loss += loss.item()
        t += (time.time() - t0)
        data_t0 = time.time()

        if it % args.log_interval == 0:
            logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * args.batch_size, len(iterator.dataset),
                    100. * it / len(iterator), loss, data_t, t,
                ))
            data_t, t = 0, 0

            if (it != 0) and args.save_batch_results:
                (epoch_test_loss, test_preds, test_labels, test_private_labels) = eval_epoch(
                    model = model, 
                    iterator = args.opt.test_generator, 
                    args = args)
                
                (epoch_valid_loss, valid_preds, valid_labels, valid_private_labels) = eval_epoch(
                    model = model, 
                    iterator = args.opt.dev_generator, 
                    args = args)

                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch=epoch+(it / len(iterator)), epochs_since_improvement=None, model=model, epoch_valid_loss=None,
                    is_best=False, 
                    )
                
                model.train()
        
    return epoch_loss / len(iterator)


# to evaluate the main model
def eval_epoch(model, iterator, args):
    
    epoch_loss = 0
    device = args.device
    
    model.eval()

    criterion = model.criterion

    preds = []
    labels = []
    private_labels = []

    for batch in iterator:
        
        text = batch[0]

        tags = batch[1]
        p_tags = batch[2]
        if len(batch) == 7:
            if not torch.is_tensor(batch[6]):
                mask = torch.stack(batch[6])
            else:
                mask = batch[6]
            mask = mask.float().to(args.device)
            if len(mask.shape) > 1:
                mask = mask.squeeze()

        text = text.to(device)
        tags = tags.to(device).long()
        p_tags = p_tags.to(device).float()

        if args.BT is not None and args.BT == "Reweighting":
            instance_weights = batch[3].float()
            instance_weights = instance_weights.to(device)

        if args.regression:
            if len(regression_tags.shape) > 1:
                regression_tags = regression_tags.squeeze()
            regression_tags = regression_tags.to(args.device)

        # main model predictions
        if args.gated:
            predictions = model(text, p_tags)
        else:
            if len(batch) == 7:
                predictions = model(text, mask)
            else:
                predictions = model(text)

        if len(predictions.shape) > 1:
            predictions = predictions if not args.regression else predictions.squeeze()

        # add the weighted loss
        if args.BT is not None and args.BT == "Reweighting":
            loss = criterion(predictions, tags if not args.regression else regression_tags)
            loss = torch.mean(loss * instance_weights)
        else:
            loss = criterion(predictions, tags if not args.regression else regression_tags)
                        
        epoch_loss += loss.item()
        
        predictions = predictions.detach().cpu()

        if args.regression:
            preds += list(predictions.numpy())
            tags = regression_tags.cpu().numpy()
        else:
            tags = tags.cpu().numpy()
            preds += list(torch.argmax(predictions, axis=1).numpy())
        labels += list(tags)

        private_labels += list(batch[2].cpu().numpy())
    
    return ((epoch_loss / len(iterator)), preds, labels, private_labels)

class BaseModel(nn.Module):

    def init_for_training(self):

        self.device = self.args.device
        self.to(self.device)

        self.learning_rate = self.args.lr
        
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.learning_rate,
            weight_decay = self.args.weight_decay,
            )

        if self.args.BT and self.args.BT == "Reweighting":
            reduction = "none"
        else:
            reduction = "mean"

        if self.args.regression:
            self.criterion = torch.nn.MSELoss(reduction = reduction)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction = reduction)
            
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2)
        
        print_network(self, verbose=True)

    def init_hyperparameters(self):
        if self.args.activation_function == "ReLu":
            self.AF = nn.ReLU()
        elif self.args.activation_function == "Tanh":
            self.AF = nn.Tanh()
        elif self.args.activation_function == "LeakyReLU":
            self.AF = nn.LeakyReLU()
        else:
            raise "not implemented yet"

        if self.args.batch_norm:
            self.BN = nn.BatchNorm1d(self.args.hidden_size)
        else:
            self.BN = None

        assert (self.args.dropout >= 0) and (self.args.dropout <= 1), "Probability must be in the range from 0 to 1"
        if self.args.dropout >= 0:
            self.dropout = nn.Dropout(p=self.args.dropout)
        else:
            self.dropout = None
    
    def zero_cls_grad(self):
        """Clears the gradients of cls layers

        """
        for group in self.cls_parameter:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
    
    def train_self(self, **opt_pairs):

        # Overwrite the arguments
        dataloader_opt_keys = ["train_generator", "dev_generator", "test_generator"]
        _generators = {k:opt_pairs.get(k, None) for k in dataloader_opt_keys}

        self.args.opt.train_generator = _generators["train_generator"] if _generators["train_generator"] is not None else self.args.opt.train_generator
        self.args.opt.dev_generator = _generators["dev_generator"] if _generators["dev_generator"] is not None else self.args.opt.dev_generator
        self.args.opt.test_generator = _generators["test_generator"] if _generators["test_generator"] is not None else self.args.opt.test_generator
        
        # Reinit the train loader for FairBatch
        if (self.args.DyBT is not None) and (self.args.DyBT in ["FairBatch", "GeneralizedFB"]):
            from .DyBT import init_sampler
            DyBT_sampler = init_sampler(self, self.args)
            # Replace the tran iterator with fairbatch version
            self.args.opt.train_generator = torch.utils.data.DataLoader(self.args.opt.train_generator.dataset, sampler=DyBT_sampler, num_workers=0)
            logging.info("Reinitialized DyBT sampler for dataloader")

        epochs_since_improvement = 0
        best_valid_loss = 1e+5
        best_valid_acc = 0
        best_valid_dto = 1e+5
        best_epoch = 0
        valid_accs, valid_fairs = [], []

        for epoch in range(self.args.opt.epochs):
            
            # Early stopping
            if epochs_since_improvement >= self.args.epochs_since_improvement:
                break
            
            # One epoch's training
            epoch_train_loss = train_epoch(
                model = self, 
                iterator = self.args.opt.train_generator, 
                args = self.args, 
                epoch = epoch)

            # One epoch's validation
            (epoch_valid_loss, valid_preds, 
            valid_labels, valid_private_labels) = eval_epoch(
                model = self, 
                iterator = self.args.opt.dev_generator, 
                args = self.args)

            # Update discriminator if needed
            if self.args.adv_debiasing and self.args.adv_update_frequency == "Epoch":
                self.args.discriminator.train_self(self)

            epoch_valid_acc = accuracy_score(valid_labels, valid_preds)
            # check DTO on validation with diff versions
            # for that purpose calc model fairness
            valid_scores, valid_confusion_matrices = gap_eval_scores(
                y_pred=valid_preds,
                y_true=valid_labels, 
                protected_attribute=valid_private_labels,
                args = self.args,
            )
            valid_accs.append(valid_scores["accuracy"])
            valid_fairs.append((1 - valid_scores["TPR_GAP"]))
            # calc vanilla dto, without any reweighting
            epoch_valid_dto = np.sqrt((1 - valid_scores["accuracy"]) ** 2 + (valid_scores["TPR_GAP"]) ** 2)
            # epoch_valid_dto = 0.0
            # Check if there was an improvement
            if self.args.early_stopping_criterion == "dto":
                is_best = epoch_valid_dto < best_valid_dto
                best_valid_dto = min(epoch_valid_dto, best_valid_dto)
            elif self.args.early_stopping_criterion == "balanced_dto":
                min_acc, max_acc = np.min(valid_accs), np.max(valid_accs)
                min_fair, max_fair = np.min(valid_fairs), np.max(valid_fairs)
                # now calc balanced dto - normalize accuracy and fairness on [0, 1] on all epochs
                balanced_fairs = (np.array(valid_fairs) - np.min(valid_fairs)) / (np.max(valid_fairs) - np.min(valid_fairs))
                balanced_accs = (np.array(valid_accs) - np.min(valid_accs)) / (np.max(valid_accs) - np.min(valid_accs))
                epoch_valid_dto = np.sqrt((1 - balanced_accs[-1]) ** 2 + (1 - balanced_fairs[-1]) ** 2)
                is_best = epoch_valid_dto < best_valid_dto
                best_epoch = epoch if is_best else best_epoch
                # we also have to renormalize best_valid_dto each time
                best_valid_dto = np.sqrt((1 - balanced_accs[best_epoch]) ** 2 + (1 - balanced_fairs[best_epoch]) ** 2)
                best_valid_dto = min(epoch_valid_dto, best_valid_dto)
            else:
                is_best = epoch_valid_loss < best_valid_loss
                best_valid_loss = min(epoch_valid_loss, best_valid_loss)
            
            #is_best = epoch_valid_acc > best_valid_acc
            #best_valid_acc = max(epoch_valid_acc, best_valid_acc)
            
            self.scheduler.step(epoch_valid_loss)

            if not is_best:
                epochs_since_improvement += 1
                logging.info("Epochs since last improvement: %d" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0
            logging.info("Loss, accuracy and DTO: %f %f %f" % (epoch_valid_loss, epoch_valid_acc, epoch_valid_dto))

            if epoch % self.args.checkpoint_interval == 0:
                logging.info("Evaluation at Epoch %d" % (epoch,))

                (epoch_test_loss, test_preds, 
                test_labels, test_private_labels) = eval_epoch(
                    model = self, 
                    iterator = self.args.opt.test_generator, 
                    args = self.args)
                
                present_evaluation_scores(
                    valid_preds, valid_labels, valid_private_labels,
                    test_preds, test_labels, test_private_labels,
                    epoch, epochs_since_improvement, self, epoch_valid_loss,
                    is_best, 
                    )

    def extract_hidden_representations(self, split):
        import numpy as np

        hidden = []
        labels = []
        private_labels = []
        regression_labels = []

        if split == "train":
            iterator = self.args.train_generator
        elif split == "dev":
            iterator = self.args.dev_generator
        elif split == "test":
            iterator = self.args.test_generator
        else:
            raise NotImplementedError

        for batch in iterator:
            
            text = batch[0]
            if len(text.shape) > 2:
                text = text.squeeze()
            tags = batch[1]
            if len(tags.shape) > 1:
                tags = tags.squeeze()
            p_tags = batch[2]
            if len(p_tags.shape) > 1:
                p_tags = p_tags.squeeze()
                
            if len(batch) == 7:
                if not torch.is_tensor(batch[6]):
                    mask = torch.stack(batch[6])
                else:
                    mask = batch[6]
                mask = mask.float().to(self.args.device)
                if len(mask.shape) > 1:
                    mask = mask.squeeze()
            
            
            labels += list(tags.cpu().numpy())
            private_labels += list(p_tags.cpu().numpy())

            text = text.to(self.args.device)
            tags = tags.to(self.args.device).long()
            p_tags = p_tags.to(self.args.device).float()

            if self.args.regression:
                regression_tags = batch[5].float()
                if len(regression_tags.shape) > 1:
                    regression_tags = regression_tags.squeeze()
                regression_labels += list(regression_tags.cpu().numpy() )
                regression_tags = regression_tags.to(self.args.device)

            # Extract hidden representations
            if self.args.gated:
                if len(batch) == 7:
                    hidden_state = self.hidden(text, mask, p_tags)
                else:
                    hidden_state = self.hidden(text, p_tags)
            else:
                if len(batch) == 7:
                    hidden_state = self.hidden(text, mask)
                else:
                    hidden_state = self.hidden(text)
            
            hidden.append(hidden_state.detach().cpu().numpy())
        
        hidden = np.concatenate(hidden, 0)

        hidden = np.array(hidden)
        labels = np.array(labels)
        private_labels = np.array(private_labels)
        regression_labels = np.array(regression_labels) if self.args.regression else None

        return hidden, labels, private_labels, regression_labels