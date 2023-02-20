import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import pandas as pd


class JigsawDataset(BaseDataset):
    embedding_type = "bert_avg_SE"
    text_type = "comment_text"

    def load_data(self):
        split_names = {"train": "train", "dev": "val", "test": "test"}
        self.filename = "jigsaw_race_{}.pq".format(split_names[self.split])

        data = pd.read_parquet(Path(self.args.data_dir) / self.filename)

        # extract and transform binary labels
        self.y = data["binary_label"]

        if self.args.encoder_architecture == "Fixed":
            self.X = list(data[self.embedding_type])
        elif self.args.encoder_architecture == "BERT":
            self.X, self.token_type_ids, self.mask = self.args.text_encoder.encoder(list(data[self.text_type]), max_length=128)
        else:
            raise NotImplementedError

        self.protected_label = data["black"].astype(np.int32) # Only race
