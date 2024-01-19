import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd
import os
import json


class SbicDataset(BaseDataset):
    embedding_type = "bert_avg_embs"
    text_type = "post"

    def load_data(self):
        #self.filename = "bios_{}_df.pkl".format(self.split)

        #data = pd.read_pickle(Path(self.args.data_dir) / self.filename)
        data = pd.read_csv(os.path.join(self.args.data_dir, f"{self.split}.csv"))
        
        if self.args.encoder_architecture == "Fixed":
            self.X = np.array(list(map(json.loads, data[self.embedding_type]))).astype(np.float32)
            self.token_type_ids, self.mask = None, None
        elif self.args.encoder_architecture == "BERT":
            self.X, self.token_type_ids, self.mask = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["targetCategoryEnc"].astype(np.float64)
        if self.args.protected_task == "gender":
            self.protected_label = data["annotatorGenderEnc"].astype(np.int32)
        elif self.args.protected_task == "race":
            self.protected_label = data["annotatorRaceEnc"].astype(np.int32)
        elif self.args.protected_task == "politics":
            self.protected_label = data["annotatorPoliticsEnc"].astype(np.int32)
        else:
            raise NotImplementedError
        
        if self.args.balance_test and self.split in ["test", "dev"]:
            classes = np.unique(self.y)
            # rebalance test set
            overall_mask_bios = np.array([False] * len(self.y))
            for class_val in classes:
                class_ids = np.where(self.y == class_val, True, False)
                class_ids = np.arange(len(self.y))[class_ids]
                # find prot_attr distribution
                vals, distr = np.unique(self.protected_label[class_ids], return_counts=True)
                min_val, min_attr = np.min(distr), np.argmin(distr)
                min_ids = class_ids[np.where(self.protected_label[class_ids] == min_attr)[0]]
                max_ids = class_ids[np.where(self.protected_label[class_ids] != min_attr)[0][:min_val]]
                #print(min_ids, class_ids)
                #print(vals,distr)
                #print(len(min_ids), len(max_ids))
                np.put(overall_mask_bios, min_ids, True)
                np.put(overall_mask_bios, max_ids, True)
            test_mask_ids = np.arange(len(overall_mask_bios))[overall_mask_bios]
            #print(np.sum(overall_mask_bios), len(self.y))
            self.X = list(np.asarray(self.X)[test_mask_ids])
            self.y = self.y[test_mask_ids]
            self.protected_label = self.protected_label[test_mask_ids]
            if self.mask is not None:
                self.mask = list(np.asarray(self.mask)[test_mask_ids])
            if self.token_type_ids is not None:
                self.token_type_ids = list(np.asarray(self.token_type_ids)[test_mask_ids])
