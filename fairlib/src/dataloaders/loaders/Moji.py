import numpy as np
from ..utils import BaseDataset
from pathlib import Path
import logging

class DeepMojiDataset(BaseDataset):

    p_aae = 0.5 # distribution of the main label, proportion of the AAE
    n = 100000 # target size

    def split_to_ids(self, texts):
        if self.split == "train":
            return texts[:40000]
        elif self.split == "dev":
            return texts[40000:42000]
        elif self.split == "test":
            return texts[42000:44000]

    def load_data(self):
        # stereotyping, 0.5 is balanced 
        if (self.split == "train" and self.args.dataset.split("_")[-1] != "balanced") or self.args.unbalance_test:
            self.ratio = 0.8 
        else:
            self.ratio = 0.5 # stereotyping, 0.5 is balanced 

        self.data_dir = Path(self.args.data_dir) / self.split

        n_1 = int(self.n * self.p_aae * self.ratio) # happy AAE 
        n_2 = int(self.n * (1-self.p_aae) * (1-self.ratio)) # happy SAE
        n_3 = int(self.n * self.p_aae * (1-self.ratio)) # unhappy AAE
        n_4 = int(self.n * (1-self.p_aae) * self.ratio) # unhappy SAE
        if self.args.unbalance_test and self.split != "train":
            # recalc n_i by real test and val sizes
            # mimic train distribution
            n_1 = n_4 = 2000
            n_2 = n_3 = int((1 - self.ratio) / self.ratio * n_1)

        if self.args.encoder_architecture == "BERT" and self.args.use_collator:
            #some test with data collator
            # in this case try to load texts
            x, token_type_ids, mask = [], [], []
            self.X = []
            for file, label, protected, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                        [1, 1, 0, 0],
                                                                        [1, 0, 1, 0], 
                                                                        [n_1, n_2, n_3, n_4]
                                                                        ):
                with open(f'{self.args.data_dir}/{file}_text', "rb") as f:
                    texts = f.readlines()
                decoded_texts = []
                import emoji
                for el in texts:
                    try:
                        # cause BERT couldn't handle emoji, transform it into text
                        if self.args.deemojify == 1:
                            decoded_texts.append(emoji.demojize(el.decode()))
                        elif self.args.deemojify == 0:
                            decoded_texts.append(el.decode())
                        else:
                            decoded_texts.append(el.decode('latin-1'))
                    except:
                        pass
                decoded_texts = self.split_to_ids(decoded_texts)
                self.X += decoded_texts[:class_n]
                self.y = self.y + [label]*len(decoded_texts[:class_n])
                self.protected_label = self.protected_label + [protected]*len(decoded_texts[:class_n])
        elif self.args.encoder_architecture == "BERT" and not self.args.use_collator:
            # in this case try to load texts
            x, token_type_ids, mask = [], [], []
            self.X, self.token_type_ids, self.mask = [], [], []
            for file, label, protected, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                        [1, 1, 0, 0],
                                                                        [1, 0, 1, 0], 
                                                                        [n_1, n_2, n_3, n_4]
                                                                        ):
                with open(f'{self.args.data_dir}/{file}_text', "rb") as f:
                    texts = f.readlines()
                decoded_texts = []
                import emoji
                for el in texts:
                    try:
                        # cause BERT couldn't handle emoji, transform it into text
                        if self.args.deemojify == 1:
                            decoded_texts.append(emoji.demojize(el.decode()))
                        elif self.args.deemojify == 0:
                            decoded_texts.append(el.decode())
                        else:
                            decoded_texts.append(el.decode('latin-1'))
                    except:
                        pass
                decoded_texts = self.split_to_ids(decoded_texts)
                buf_x, buf_token_type_ids, buf_mask = self.args.text_encoder.encoder(decoded_texts)
                self.X += buf_x[:class_n]
                self.token_type_ids += buf_token_type_ids[:class_n]
                self.mask += buf_mask[:class_n]
                self.y = self.y + [label]*len(buf_x[:class_n])
                self.protected_label = self.protected_label + [protected]*len(buf_x[:class_n])
        else:
            for file, label, protected, class_n in zip(['pos_pos', 'pos_neg', 'neg_pos', 'neg_neg'],
                                                                        [1, 1, 0, 0],
                                                                        [1, 0, 1, 0], 
                                                                        [n_1, n_2, n_3, n_4]
                                                                        ):
                data = np.load('{}/{}.npy'.format(self.data_dir, file))
                data = list(data[:class_n])
                self.X = self.X + data
                self.y = self.y + [label]*len(data)
                self.protected_label = self.protected_label + [protected]*len(data)