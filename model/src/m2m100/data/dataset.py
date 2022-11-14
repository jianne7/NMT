
import os
import torch
from torch.utils.data import Dataset

from itertools import chain
import sys
import pandas as pd
import json

from pprint import pprint

from tqdm import tqdm

def load_json(file_path):

    with open(file_path) as f:
        data = json.load(f)

    return data


class TranslationDataset(Dataset):
    def __init__(self, file_path, mode, tokenizer):
        self.file_path = file_path
        self.mode = mode
        self.tokenizer = tokenizer
        
        # preprocess data
        self.preprocess()

    def preprocess(self):
        data = load_json(self.file_path)

        self.examples = []

        for item in tqdm(data):
            for b in item['Body']:
                id = b['ID']

                src_lang = b['Data']['Source_Lang']
                src_text = b['Data']['Text']

                if b['Data']['Translated']['en'] is not None:
                    tgt_lang = 'en'
                    tgt_text = b['Data']['Translated']['en']['Original']['Text']
                elif b['Data']['Translated']['zh'] is not None:
                    tgt_lang = 'zh'
                    tgt_text = b['Data']['Translated']['zh']['Original']['Text']
                elif b['Data']['Translated']['ja'] is not None:
                    tgt_lang = 'ja'
                    tgt_text = b['Data']['Translated']['ja']['Original']['Text']
                else:
                    pass

                # print(f"({id:05d}) src_text: {src_text}")
                # print(f"({id:05d}) tgt_text: {tgt_text}")
                # print("="*80)

                self.examples.append((src_lang, src_text, tgt_lang, tgt_text))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        src_lang, src_text, tgt_lang, tgt_text = self.examples[index]

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        example = self.tokenizer(src_text, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt_text, truncation=True).input_ids
            example['labels'] = labels

        return example

