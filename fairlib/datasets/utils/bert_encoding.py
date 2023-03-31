import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import pickle
from tqdm import tqdm

class BERT_encoder:
    def __init__(self, batch_size=128, model_name='bert-base-cased') -> None:
        self.batch_size = batch_size
        self.model, self.tokenizer = self.load_lm(model_name=model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

    
    def load_lm(self, model_name):
        model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, model_name)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        return model, tokenizer

    def tokenize(self, data):

        tokenized_data = []
        attention_mask = []
        total_n = len(data)
        n_iterations = (total_n // self.batch_size) + (total_n % self.batch_size > 0)
        for i in tqdm(range(n_iterations)):
            row_lists = list(data)[i*self.batch_size:(i+1)*self.batch_size]
            tokens = self.tokenizer(row_lists, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens['input_ids']
            masks = tokens['attention_mask']
            attention_mask.append(masks)
            tokenized_data.append(input_ids)
        return tokenized_data, attention_mask

    def encode_text(self, data, masks):
        all_data_cls = []
        all_data_avg = []
        for row, mask in tqdm(zip(data, masks)):
            with torch.no_grad():
                input_ids = row.to(self.device)
                mask = mask.to(self.device)
                last_hidden_states = self.model(input_ids, mask)[0].detach().cpu()
                all_data_avg.append(last_hidden_states.mean(dim=1).numpy())
                all_data_cls.append(last_hidden_states[:,0].numpy())
                input_ids = input_ids.detach().cpu()
        return np.vstack(np.array(all_data_avg)), np.vstack(np.array(all_data_cls))


    def encode(self, data):
        tokens, masks = self.tokenize(data)

        avg_data, cls_data = self.encode_text(tokens, masks)

        return avg_data, cls_data