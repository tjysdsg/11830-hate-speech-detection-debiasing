import pandas as pd
from .common import *
from torch.utils.data import DataLoader, Dataset
import torch


class FountaProcessor(DataProcessor):
    """
    Data processor using DataProcessor class provided by BERT
    """

    def __init__(self, configs, tokenizer=None):
        super().__init__()
        self.data_dir = configs.data_dir
        self.tokenizer = tokenizer
        self.max_seq_length = configs.max_seq_length
        self.configs = configs
        self.remove_nw = configs.remove_nw

    def _get_neutral_word_ids(self):
        f = open(self.configs.neutral_words_file)
        neutral_words = []
        neutral_words_ids = set()
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            canonical = self.tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            word = canonical[0]
            neutral_words.append(word)
            neutral_words_ids.add(self.tokenizer.vocab[word])
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words

    def _create_examples(self, data_dir, split, label=None):
        df = pd.read_csv(os.path.join(data_dir, f'founta_{split}_clean.csv'))
        examples = []
        for i, row in df.iterrows():
            text = row['CleanText']
            example = InputExample(
                text_a=text, guid=f'{split}-{i}',
                label=int(row['Label'])
            )

            if label is None or example.label == label:
                examples.append(example)

        return examples

    def get_train_examples(self, data_dir, **kwargs):
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir, **kwargs):
        return self._create_examples(data_dir, 'valid')

    def get_test_examples(self, data_dir, **kwargs):
        return self._create_examples(data_dir, 'test')

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0, 1]

    def get_features(self, split):
        if split == 'dev':
            split = 'valid'
        neutral_word_ids = []
        if self.configs.remove_nw:
            neutral_word_ids = self._get_neutral_word_ids()

        examples = self._create_examples(self.data_dir, split)
        features = []
        for example in examples:
            tokens = self.tokenizer.tokenize(example.text_a)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if self.configs.remove_nw:
                input_ids = list(filter(lambda x: x not in neutral_word_ids, input_ids))
            length = len(input_ids)
            padding = [0] * (self.max_seq_length - length)
            input_ids += padding
            input_ids = torch.LongTensor(input_ids)
            features.append({'text': input_ids, 'length': length})
        return features

    def get_dataloader(self, split, batch_size=1):
        features = self.get_features(split)
        dataset = FountaDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dotdict_collate)
        return dataloader

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class FountaDataset(Dataset):
    """
    torch.utils.Dataset instance for building torch.utils.DataLoader, for training the language model.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)
