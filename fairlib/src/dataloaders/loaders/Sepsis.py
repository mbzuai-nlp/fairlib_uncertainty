import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
from functools import reduce

gender2id = {
    "M":0,
    "F":1,
}

ethnicity2id = {'White': 0, 
           'Other': 1, 
           'Black': 2, 
           'Hispanic': 3, 
           'Asian': 4}

class SepsisDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "x"

    def load_data(self):
        self.filename = "sepsis_sex_{}_df.pkl".format(self.split)
        if self.args.protected_task == "ethnicity":
            self.filename = "sepsis_ethnicity_{}_df.pkl".format(self.split)
        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X, self.token_type_ids, self.mask = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["label"].astype(np.float64) #Profession
        if self.args.protected_task == "sex":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "ethnicity":
            self.protected_label = data["ethnicity_class"].astype(np.int32) # Ethnicity
            
            
        if self.args.subsample_protected_labels:
            dev_data = pd.read_pickle(Path(self.args.data_dir) / self.filename.replace(self.split, "dev"))
            dev_y = dev_data["label"].astype(np.float64)
            if self.args.protected_task == "sex":
                dev_protected_label = dev_data["gender_class"].astype(np.int32) # Gender
            elif self.args.protected_task == "ethnicity":
                dev_protected_label = dev_data["ethnicity_class"].astype(np.int32) # Ethnicity
            
            classes = np.unique(dev_y)
            large_attrs_cls = []
            for c in classes:
                value_counts = dev_protected_label[dev_y == c].value_counts() 
                large_attrs_cls.append(value_counts[value_counts > 20].index)
            large_attrs = reduce(np.intersect1d, large_attrs_cls)
            
            ids = np.isin(self.protected_label, large_attrs)
            
            self.X = list(np.asarray(self.X)[ids])
            self.y = self.y[ids].reset_index(drop=True)
            self.protected_label = self.protected_label[ids].reset_index(drop=True)
            if self.mask is not None:
                self.mask = list(np.asarray(self.mask)[ids])
            if self.token_type_ids is not None:
                self.token_type_ids = list(np.asarray(self.token_type_ids)[ids])
            
        if self.args.balance_test and self.split in ["test", "dev"]:  
            classes = np.unique(self.y)
            attrs = np.unique(self.protected_label)
            # rebalance test set
            overall_mask_bios = np.array([False] * len(self.y))
            for class_val in classes:
                class_ids = np.where(self.y == class_val, True, False)
                class_ids = np.arange(len(self.y))[class_ids]
                # find prot_attr distribution
                vals, distr = np.unique(self.protected_label[class_ids], return_counts=True)
                min_val, min_attr = np.min(distr), np.argmin(distr)
                min_val = max(min_val, 20)
                min_ids = class_ids[np.where(self.protected_label[class_ids] == min_attr)[0]]
                np.put(overall_mask_bios, min_ids, True)
                for attr in attrs:
                    if attr == min_attr:
                        continue
                    idx = self.protected_label[class_ids] == attr
                    if min_val < idx.sum(): 
                        max_ids = class_ids[np.where(idx)[0][:min_val]]
                    else:
                        max_ids = class_ids[np.where(idx)[0]]
                    np.put(overall_mask_bios, max_ids, True)
            test_mask_ids = np.arange(len(overall_mask_bios))[overall_mask_bios]
            self.X = list(np.asarray(self.X)[test_mask_ids])
            self.y = self.y[test_mask_ids]
            self.protected_label = self.protected_label[test_mask_ids]
            if self.mask is not None:
                self.mask = list(np.asarray(self.mask)[test_mask_ids])
            if self.token_type_ids is not None:
                self.token_type_ids = list(np.asarray(self.token_type_ids)[test_mask_ids])