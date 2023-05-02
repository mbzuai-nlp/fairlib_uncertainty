import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

gender2id = {
    "m":0,
    "f":1
}

class BiosDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "hard_text"

    def load_data(self):
        self.filename = "bios_{}_df.pkl".format(self.split)

        data = pd.read_pickle(Path(self.args.data_dir) / self.filename)
        if self.args.subsample_classes is not None:
            # use only selected columns from target class
            data = data[data["p"].isin(self.args.subsample_classes)]
            # after reindex target labels
            data = data.reset_index(drop=True)
            old_targets = data["profession_class"].unique()
            old_targets.sort()
            map_new_targets = {key: value for key, value in zip(old_targets, range(len(old_targets)))}
            data["profession_class"].replace(map_new_targets, inplace=True)

        if self.args.protected_task in ["economy", "both"] and self.args.full_label:
            selected_rows = (data["economy_label"] != "Unknown")
            data = data[selected_rows]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
            self.token_type_ids, self.mask = None, None
        elif self.args.encoder_architecture == "BERT":
            self.X, self.token_type_ids, self.mask = self.args.text_encoder.encoder(list(data[self.text_type]))
        else:
            raise NotImplementedError

        self.y = data["profession_class"].astype(np.float64) #Profession
        if self.args.protected_task == "gender":
            self.protected_label = data["gender_class"].astype(np.int32) # Gender
        elif self.args.protected_task == "economy":
            self.protected_label = data["economy_class"].astype(np.int32) # Economy
        else:
            self.protected_label = data["intersection_class"].astype(np.int32) # Intersection
        if self.args.subsample_all < 1.0:
            # subsample split
            subsample_len = int(len(self.y) * self.args.subsample_all)
            self.X = self.X[:subsample_len]
            self.y = self.y[:subsample_len]
            self.protected_label = self.protected_label[:subsample_len]
            if self.mask is not None:
                self.mask = self.mask[:subsample_len]
            if self.token_type_ids is not None:
                self.token_type_ids = self.token_type_ids[:subsample_len]
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
