import os
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm
import random
import time
import logging

import yaml

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from src.m2m100.data.dataset import TranslationDataset


torch.set_printoptions(precision=8, sci_mode=False)


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_device(ngpu: int, logger) -> torch.device:
    use_cuda = True if ngpu > 0 else False

    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    logger.info(f"device: {device}")

    if str(device) == 'cuda':
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

        for idx in range(torch.cuda.device_count()):
            logger.info("device(%d) : %s" % (idx, torch.cuda.get_device_name(idx)))
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("CUDA version : %s" % torch.version.cuda)
        logger.info("PyTorch version : %s" % torch.__version__)
    else:
        logger.info("CUDA is available : %s" % (torch.cuda.is_available()))
        logger.info("PyTorch version : %s" % torch.__version__)

    return device


def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def make_directory(directory: str) -> str:
    """경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지
    """

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"

        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg


def pad(data, pad_id, max_len):
    padded_data = list(map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]), data))
    return padded_data

def collate_fn_translation(batch):
    features = {}

    input_ids = [torch.LongTensor(item['input_ids']) for item in batch]
    max_len_input_ids = max(list(map(lambda x: len(x), input_ids)))
    input_ids_padded = pad(input_ids, pad_id=1, max_len=max_len_input_ids)
    features['input_ids'] = torch.stack(input_ids_padded, dim=0).type(torch.LongTensor)
    # print(features)

    attention_mask = [torch.Tensor(item['attention_mask']) for item in batch]
    max_len_attention_mask = max(list(map(lambda x: len(x), attention_mask)))
    attention_mask_padded = pad(attention_mask, pad_id=0, max_len=max_len_attention_mask)
    features['attention_mask'] = torch.stack(attention_mask_padded, dim=0).type(torch.Tensor)
    # print(features)
    
    labels = [torch.LongTensor(item['labels']) for item in batch]
    max_len_labels = max(list(map(lambda x: len(x), labels)))
    labels_padded = pad(labels, pad_id=-100, max_len=max_len_labels)
    features['labels'] = torch.stack(labels_padded, dim=0).type(torch.LongTensor)
    # print(features)

    return features


if __name__ == '__main__':
    # Logging
    make_directory('exp')
    logger = get_logger(name='train',
                file_path=os.path.join('exp', 'train_log.log'), stream=True)


    # Config & Seed & Device
    config = load_yaml("./config/train_config.yaml")
    logger.info(f"Config : {config}")

    set_random_seed(config['random_seed'])

    device = get_device(ngpu=1, logger=logger)


    # Load dataset & dataloader
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M", src_lang="ko", tgt_lang="en")
    train_dataset = TranslationDataset(file_path="data/ko2en_travel_1_training_Bleu_Grouge.json",
                        mode='train',
                        tokenizer=tokenizer)    

    print(len(train_dataset), train_dataset[0])

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(train_dataset, 
                                shuffle=True,
                                batch_size=2, # 4 
                                collate_fn=collate_fn_translation)
    
    valid_dataloader = DataLoader(train_dataset, 
                                shuffle=False,
                                batch_size=1, 
                                collate_fn=collate_fn_translation)
    
    
    for batch in train_dataloader:
        print({k: v.size() for k, v in batch.items()})
        break


    
    # Model

    from transformers import M2M100ForConditionalGeneration

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model.to(device)

    from transformers import AdamW

    optimizer = AdamW(model.parameters(), lr=5e-5)

    from transformers import get_scheduler

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print(num_training_steps)



    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            logger.info(f"loss: {loss.item():.8f}")

    
    model.eval()
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        ref_text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        del batch['labels']
        generated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id("en"))
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        logger.info(f"ref: {ref_text}")
        logger.info(f"hyp: {translated_text}")
        break
    
    model.save_pretrained('voiceprint/m2m100_418M')
    tokenizer.save_pretrained('voiceprint/m2m100_418M') 
