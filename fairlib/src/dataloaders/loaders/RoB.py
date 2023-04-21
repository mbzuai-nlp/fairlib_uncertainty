import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

gender2id = {
    "Male":0,
    "Female":1,
    "Male and Female":2,
}

area2id = {
    'Child health': 0,
    'Heart & circulation': 1,
    'Lungs & airways': 2,
    'Mental health': 3,
    'Gastroenterology & hepatology': 4,
    'Cancer': 5,
    'Insurance medicine': 6,
    'Infectious disease': 7,
    'Neurology': 8,
    'Complementary & alternative medicine': 9,
    'Pain & anaesthesia': 10,
    'Gynaecology': 11,
    'Tobacco, drugs & alcohol': 12,
    'Kidney disease': 13,
    'Rheumatology': 14,
    'Pregnancy & childbirth': 15,
    'Orthopaedics & trauma': 16,
    'Endocrine & metabolic': 17,
    'Dentistry & oral health': 18,
    'Effective practice & health systems': 19,
    'Eyes & vision': 20,
    'Blood disorders': 21,
    'Urology': 22,
    'Developmental, psychosocial & learning problems': 23,
    'Skin disorders': 24,
    'Neonatal care': 25,
    'Health & safety at work': 26,
    'Genetic disorders': 27,
    'Ear, nose & throat': 28,
    'Consumer & communication strategies': 29,
    'Wounds': 30,
    'Public health': 31,
    'Allergy & intolerance': 32
}

class RoBDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "x"

    def load_data(self):
        self.filename = "rob_{}_df.pkl".format(self.split)
        if self.args.protected_task == "area":
            self.filename = "rob_area_{}_df.pkl".format(self.split)
        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X, self.token_type_ids, self.mask = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["y"].astype(np.float64) #Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "area":
            self.protected_label = data["area_class"].astype(np.int32) # Gender
            
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