import numpy as np

import torch
from torch import nn

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score

from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AdamW

from dataset import TranslationDataset
import warnings
import gc

warnings.filterwarnings(action='ignore')
gc.collect()


torch.cuda.empty_cache()


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
    # features['labels'] = torch.stack(labels_padded, dim=0).type(torch.LongTensor)
    features['labels'] = torch.stack(labels_padded, dim=0).type(torch.LongTensor)

    # print(features)

    return features


def evaluate(model, tokenizer, device):
    # train_dataset = TranslationDataset(file_path='data/sns100.csv',
    #                                     mode='train',
    #                                     tokenizer=tokenizer)
    # eval_dataloader = DataLoader(train_dataset, 
    #                              shuffle=False,
    #                              batch_size=1, 
    #                              collate_fn=collate_fn_translation)
    
    dataset = TranslationDataset(file_path='data/sns100.csv',
                                 tokenizer=tokenizer)
    dataloader = DataLoader(dataset, 
                            shuffle=False,
                            batch_size=1, 
                            collate_fn=collate_fn_translation)

    bleu_scores = []
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        ref_text = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        del batch['labels']
        generated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id("en"), num_beams=1)
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # print(f"ref: {ref_text}")
        # print(f"hyp: {translated_text}")

        candi = translated_text[0].split(' ')
        ref = ref_text[0].split(' ')
    
        bscore = bleu_score([candi], [[ref]], max_n=1, weights=[1])
        # print(f'blue_score : {bscore}')
        bleu_scores.append(bscore)
        #break
        
    results = np.mean(bleu_scores)

    return results


def train_teacher(num_epochs, device, teacher_model, tokenizer):
    # device = torch.device('cpu')
    # teacher_model = model.to(device)
    # #criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(teacher_model.parameters(), lr=5e-5)

    train_dataset = TranslationDataset(file_path='data/ko2en_travel_1_training_Bleu_Grouge.json',
                        mode='train',
                        tokenizer=tokenizer)

    train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=2,
            shuffle=True,
            #num_workers=1,
            collate_fn=collate_fn_translation
        )

    num_epochs = num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        teacher_model.train()
        losses = []

        pbar = tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=True, desc=f'Epoch {epoch}')
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = teacher_model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            # break
        # break

        try:
            avg_loss = sum(losses) / len(losses)
        except ZeroDivisionError:
            avg_loss = 0
        acc = evaluate(teacher_model, tokenizer, device)
        print(f'Loss:{avg_loss:.2f}\tBleu:{acc:.2f}')



    return teacher_model


def train_step(teacher_model, student_model, optimizer, divergence_loss_fn, temp, alpha, epoch, device):
#     train_dataset = TranslationDataset(file_path='data/ko2en_travel_1_training_Bleu_Grouge.json',
#                         mode='train',
#                         tokenizer=tokenizer)

#     train_dataloader = DataLoader(
#             dataset=train_dataset, 
#             batch_size=2,
#             shuffle=True,
#             #num_workers=1,
#             collate_fn=collate_fn_translation
#         )
    
    dataset = TranslationDataset(file_path='data/sns100.csv',
                                        tokenizer=tokenizer)

    dataloader = DataLoader(
                            dataset=dataset, 
                            batch_size=1,
                            shuffle=True,
                            #num_workers=1,
                            collate_fn=collate_fn_translation
                        )
    losses = []
    teacher_ppls = []
    student_ppls = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        teacher_model.eval()
        student_model.train()
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**batch)
            teacher_logits = teacher_outputs.logits
            teacher_ppls.append(torch.exp(teacher_outputs.loss))
                
        student_outputs = student_model(**batch)
        student_loss = student_outputs.loss
        student_logits = student_outputs.logits
        student_ppls.append(torch.exp(student_loss))

        max_vocab_size = teacher_logits.shape[-1]
        distillation_loss = divergence_loss_fn(
                                                F.log_softmax(student_logits.view(-1,max_vocab_size) / temp, dim=1),
                                                F.softmax(teacher_logits.view(-1, max_vocab_size) / temp, dim=1)
                                            )
        loss = alpha * student_loss + (1 - alpha) * distillation_loss
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    avg_loss = sum(losses) / len(losses)
    teacher_avg_ppls = sum(teacher_ppls) / len(teacher_ppls)
    student_avg_ppls = sum(student_ppls) / len(student_ppls)

    return avg_loss, teacher_avg_ppls, student_avg_ppls


def main(teacher_model, student_model, tokenizer, device, temp=7, alpha=0.3):
    # teacher_model = teacher_model.to(device)
    # student_model = student_model.to(device)
    # student_loss_fn = nn.CrossEntropyLoss()
    divergence_loss_fn = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        # loss = train_step(teacher_model, student_model, optimizer, student_loss_fn, divergence_loss_fn, temp, alpha, epoch, device)
        # acc = check_accuracy(test_loader, student, device)
        loss, teacher_ppl, student_ppl = train_step(teacher_model, student_model, optimizer, divergence_loss_fn, temp, alpha, epoch, device)
        bleu_score = evaluate(student_model, tokenizer, device)
        print(f'Loss : {loss:.2f}\tTeacher_ppl : {teacher_ppl:.2f}\tStudent_ppl : {student_ppl:.2f}\tBleu_score : {bleu_score:.2f}')


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'voiceprint/m2m100_418M'

    teacher_model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    teacher_model.to(device)

    student_model = M2M100ForConditionalGeneration.from_pretrained(model_path, encoder_layers=6, decoder_layers=6)
    student_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    main(teacher_model, student_model, tokenizer, device, temp=7, alpha=0.3)