from fairlib.datasets.utils.download import download
from fairlib.datasets.utils.bert_encoding import BERT_encoder
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path

gender2id = {
    "M":0,
    "F":1,
}

ethnicity2id = {'White': 0, 
           'Other': 1, 
           'Black': 2, 
           'Hispanic': 3, 
           'Asian': 4}


class2id = {'non_septic': 0, 'septic': 1}

class Sepsis:

    _NAME = "sepsis"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.encoder = BERT_encoder(self.batch_size)

    def bert_encoding(self):
        for split in self._SPLITS:
            split_df = pd.DataFrame(pd.read_json(Path(self.dest_folder)/"{}.json".format(split)))

            text_data = list(split_df["x"])
            avg_data, cls_data = self.encoder.encode(text_data)
            split_df["bert_avg_SE"] = list(avg_data)
            split_df["bert_cls_SE"] = list(cls_data)
            
            if "sex" in self.dest_folder:
                split_df["gender_class"] = split_df["protected_label"].map(gender2id)
            else:
                split_df["ethnicity_class"] = split_df["protected_label"].map(ethnicity2id)
            split_df["label"] = split_df["y"].map(class2id)

            if "sex" in self.dest_folder:
                split_df.to_pickle(Path(self.dest_folder) / "sepsis_sex_{}_df.pkl".format(split))
            else:
                split_df.to_pickle(Path(self.dest_folder) / "sepsis_ethnicity_{}_df.pkl".format(split))

    def prepare_data(self):
        self.bert_encoding()