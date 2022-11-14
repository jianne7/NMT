import pandas as pd

import torch
from torchtext.data.metrics import bleu_score
from sacrebleu.metrics import BLEU
from sacrebleu.tokenizers.tokenizer_ter import TercomTokenizer


def read_csv(path): #column_name : str):
    data = pd.read_csv(path, encoding='utf-8-sig')
    # data = data[column_name]
    # data = data.tolist()
    return data


def torch_bleu(ref_txt : str, candi_txt : str, tgt_lang : str):

    if tgt_lang == 'ko' or tgt_lang == "zh" or tgt_lang == "ja":
        ref_txt = list(ref_txt.strip().replace(' ', ''))
        candi_txt = list(candi_txt.strip().replace(' ', ''))
    else:
        ref_txt = ref_txt.split()
        candi_txt = candi_txt.split() 
    # ref_txt = ref_txt.split() if tgt_lang == 'en' else list(ref_txt.strip().replace(' ', ''))
    # candi_txt = candi_txt.split() if tgt_lang == 'en' else list(candi_txt.strip().replace(' ', ''))
    print(f'ref_txt : {ref_txt}')
    print(f'candi_txt : {candi_txt}')
    # print(f'len ref_txt : {len(ref_txt)}')
    # print(f'len candi_txt : {len(candi_txt)}')
    bscore = bleu_score([candi_txt], [[ref_txt]], max_n=1, weights=[1])
    print(f'bleu_score: {bscore}')

    return bscore


def sacre_bleu(ref_txt : str, candi_txt : str, lang : str):
    print(f'ref_txt : {ref_txt}')
    print(f'candi_txt : {candi_txt}')
    tokenizer = TercomTokenizer(normalized=True, asian_support=True)
    # if  lang == "en":
    #     bleu = BLEU()
    if lang =='zh':
        bleu = BLEU(tokenize='zh', smooth_method="add-k")  # smooth method 기존 exp에서 add-k로 변경 k=1로 사실상 add-1
    elif lang == 'ja':
        bleu = BLEU(tokenize='ja-mecab', smooth_method="add-k")
    elif lang == "ko":
        bleu = BLEU(tokenize="char", smooth_method="add-k")
    else:
        bleu = BLEU(smooth_method="add-k")

    bscore = (bleu.corpus_score([candi_txt.strip()], [[ref_txt.strip()]]).score) * 0.01
    print(f'bleu_score: {bleu.corpus_score([candi_txt], [[ref_txt]])}')

    return bscore


if __name__ == '__main__':
    file_path = "/Users/ujinne/python/Lamp/m2m100/bleuscore/valid.csv"
    data = read_csv(file_path)
    ref_txt = data['tgt_text'].tolist()
    candi_col = input('input the col_name: ')
    candi_txt = data[candi_col].tolist()
    lang = data['tgt_lang'].tolist()
    # lang = 'en', 'ja', 'zh'
    # lang = 'zh'

    # # papago = data['papago'].tolist()
    # m2m = data['m2m'].tolist()
    # english_centric = data['english_centric'].tolist()
    # ref_txt = read_csv(file_path, 'original')
    # google = read_csv(file_path, 'google')
    # m2m = read_csv(file_path, 'm2m')
    # english_centric = read_csv(file_path, 'english_centric')


    torchbleu = []
    sacrebleu = []
    for i in range(len(ref_txt)):
        print(f'i={i}')
        try:
            print("------torch_bleu-----")
            torch_bs = torch_bleu(ref_txt[i], candi_txt[i], lang[i])
            print("------sacre_bleu-----")
            sacre_bs = sacre_bleu(ref_txt[i], candi_txt[i], lang[i])
        except:
            print("------torch_bleu-----")
            torch_bs = 0
            print("------sacre_bleu-----")
            sacre_bs = 0
        
        torchbleu.append(torch_bs)
        sacrebleu.append(sacre_bs)
        # try:    
        #     print("------torch_bleu-----")
        #     torch_bs = torch_bleu(ref_txt[i], candi_txt[i])
        #     torchbleu.append(torch_bs)
        # except:
        #     torchbleu.append("ERR")
        
        # try:
        #     print("------sacre_bleu-----")
        #     sacre_bs = sacre_bleu(ref_txt[i], candi_txt[i])
        #     sacrebleu.append(sacre_bs)
        # except:
        #     sacrebleu.append("ERR")     
        print("="*100)
        # print(torchbleu.count("ERR"))
        # print(sacrebleu.count("ERR"))

    torch_bs = pd.Series(torchbleu, name = candi_col+'_tbs')
    sacre_bs = pd.Series(sacrebleu, name = candi_col+'_sbs')

    result = pd.concat([data, torch_bs, sacre_bs], axis=1)
    result.to_csv(file_path, index=False, encoding="utf-8-sig")




