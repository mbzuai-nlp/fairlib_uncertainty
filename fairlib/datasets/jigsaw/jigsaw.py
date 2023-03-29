from fairlib.datasets.utils.bert_encoding import BERT_encoder
import numpy as np
import pandas as pd
import os
from pathlib import Path
import gdown


class Jigsaw:

    _NAME = "Jigsaw"
    _SPLITS = ["train", "val", "test"]

    def __init__(self, dest_folder, batch_size):
        self.dest_folder = dest_folder
        self.batch_size = batch_size
        self.encoder = BERT_encoder(self.batch_size)

    def download_files(self):
        folder_id = "1v0QzeIgVbjdxCv4EDz4Joo4lGH4MKBom"
        gdown.download_folder(id=folder_id, output=self.dest_folder, quiet=True)

    def bert_encoding(self):
        for split in self._SPLITS:
            split_df = pd.DataFrame(pd.read_parquet(Path(self.dest_folder)/"jigsaw_race_{}.pq".format(split)))

            text_data = list(split_df["comment_text"])
            avg_data, cls_data = self.encoder.encode(text_data)
            split_df["bert_avg_SE"] = list(avg_data)
            split_df["bert_cls_SE"] = list(cls_data)
            split_df["binary_label"] = np.where(split_df["toxicity"] > 0.5, 1, 0)

            split_df.to_parquet(Path(self.dest_folder) / "jigsaw_race_{}.pq".format(split))

    def prepare_data(self):
        self.download_files()
        self.bert_encoding()
