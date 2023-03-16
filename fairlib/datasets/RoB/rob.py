from fairlib.datasets.utils.download import download
from fairlib.datasets.utils.bert_encoding import BERT_encoder
from fairlib.src.utils import seed_everything
import numpy as np
import pandas as pd
import os
from pathlib import Path

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

class RoB:

    _NAME = "RoB"
    _SPLITS = ["train", "dev", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.encoder = BERT_encoder(self.batch_size)

    def bert_encoding(self):
        for split in self._SPLITS:
            split_df = pd.DataFrame(pd.read_json(Path(self.dest_folder)/"{}_noncompact.json".format(split)))

            text_data = list(split_df["x"])
            avg_data, cls_data = self.encoder.encode(text_data)
            split_df["bert_avg_SE"] = list(avg_data)
            split_df["bert_cls_SE"] = list(cls_data)
            if "gender" in self.dest_folder:
                split_df["gender_class"] = split_df["protected_label"].map(gender2id)
            else:
                split_df["area_class"] = split_df["protected_label"].map(area2id)
            split_df["label"] = split_df["y"]

            if "gender" in self.dest_folder:
                split_df.to_pickle(Path(self.dest_folder) / "rob_{}_df.pkl".format(split))
            else:
                split_df.to_pickle(Path(self.dest_folder) / "rob_area_{}_df.pkl".format(split))

    def prepare_data(self):
        self.bert_encoding()