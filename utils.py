import io
import torch
from collections import Counter
from torchtext.vocab import Vocab


# process data
def get_sentences(filepaths):
    lang1_sentences = open(filepaths[0]).readlines()
    lang2_sentences = open(filepaths[1]).readlines()
    return lang1_sentences, lang2_sentences


class Vocabulary(object):
    def __init__(self, tokenizer, freq_threshold=2):
        self.stoi = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.tokenizer = tokenizer
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.stoi)

    def tokenize_str(self, text):
        return [tok.lower() for tok in self.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            for token in self.tokenize_str(sentence):
                if token not in self.stoi:
                    index = len(self.stoi)
                    self.stoi[token] = index
                    self.itos[index] = token

    def numericalize(self, text):
        tokenized_text = self.tokenize_str(text)

        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]
