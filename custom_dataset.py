import os  # loading files
import pandas as pd  # annotation

import torch
from torch.nn.utils.rnn import pad_sequence  # for padding
from torch.utils.data import Dataset  # for loading data


class Seq2SeqDataset(Dataset):
    def __init__(self, lang1_sentences, lang2_sentences, lang1_vocab, lang2_vocab):
        # load data
        self.lang1_sentences = lang1_sentences
        self.lang2_sentences = lang2_sentences

        self.lang1_vocab = lang1_vocab
        self.lang2_vocab = lang2_vocab

    def __len__(self):
        return len(self.lang1_sentences)

    def __getitem__(self, index):

        text1 = self.lang1_sentences[index]
        numericalized_text1 = [self.lang1_vocab.stoi['<BOS>']]
        numericalized_text1 += self.lang1_vocab.numericalize(text1)
        numericalized_text1.append(self.lang1_vocab.stoi['<EOS>'])

        text2 = self.lang2_sentences[index]
        numericalized_text2 = [self.lang2_vocab.stoi['<BOS>']]
        numericalized_text2 += self.lang2_vocab.numericalize(text2)
        numericalized_text2.append(self.lang2_vocab.stoi['<EOS>'])

        return torch.tensor(numericalized_text1), torch.tensor(numericalized_text2)

    def generate_batch(self, data_batch):
        '''
        :param data_batch:
        :return: tuple of batches for each language in format Batch X Sequence Length
        '''
        lang1_batch, lang2_batch = [], []
        for (lang1_item, lang2_item) in data_batch:
            lang1_batch.append(lang1_item)
            lang2_batch.append(lang2_item)
        lang1_batch = pad_sequence(lang1_batch, padding_value=self.lang1_vocab.stoi['<PAD>'])
        lang2_batch = pad_sequence(lang2_batch, padding_value=self.lang2_vocab.stoi['<PAD>'])
        return lang1_batch, lang2_batch
