from transformers import AutoTokenizer

class text2id():
    """mapping natural language to numeric identifiers.
    """
    def __init__(self, args) -> None:
        if args.encoder_architecture == "Fixed":
            self.encoder = None
        elif args.encoder_architecture == "BERT":
            self.model_name = args.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            raise NotImplementedError
    
    def encoder(self, sample, max_length=128):
        encodings = self.tokenizer(sample, max_length=max_length, truncation=True, padding='max_length')
        return encodings["input_ids"], encodings['token_type_ids'], encodings['attention_mask']