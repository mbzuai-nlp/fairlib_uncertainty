import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd

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
            self.protected_label = data["ethnicity_class"].astype(np.int32) # Gender