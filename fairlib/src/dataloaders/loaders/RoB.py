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