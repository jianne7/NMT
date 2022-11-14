import os
import re
import json
import numpy as np
import parmap

from tqdm import tqdm
from Language_detector import LangDetector
# from multiprocessing import Pool


# 동북 아시아
ko = '\u3131-\u3163|\uac00-\ud7af|\u0041-\u007a'    # ko + en
zh = '\u4e00-\u62ff|\u6300-\u77ff|\u7800-\u8cff|\u8d00-\u9fff|\u0041-\u007a' # zh_cn + en
ja = '\u3041-\u3096|\u30a0-\u30ff|\u3400-\u4db5|\u4e00-\u9fcb|\uf900-\ufa6a|\u0041-\u007a'  #ja + en

# 러시아
ru = '\u0410-\u044f'                 # ru

# 유럽 / 미국 (vi, de, pt, fr, id, es, it, en)
for country in 'vi,de,pt,fr,id,es,it,en'.split(','):
    globals()[f'{country}'] = '\u0041-\u007a|\u00c0-\u0178|\u1e00-\u1eff|\u0180-\u024f|\u1e00—\u1eff|\u0027'

# 인도
hi = '\u0900-\u097f|\ua8e0-\ua8ff' # hi (드 파나 가리어)

# 아랍
ar = '\u0627-\u064a'    

# 필리핀어
tl = '\u1700-\u171f|\u0041-\u007a|\u00c0-\u0178|\u1e00-\u1eff|\u0180-\u024f|\u1e00—\u1eff|\u0027'


# src language 설정
SRC_UNICODE = vi
SRC_LANGCODE = "vi"


def save_json(path: str, json_inputs: dict):
    with open(path, "w", encoding="utf-8-sig") as file:
        file.write(json.dumps(json_inputs, indent='\t', ensure_ascii=False))


def clean_lng(sentence, unicode):
    '''
    문장 전처리 함수
    문장과 해당 언어의 유니코드를 문자열로 넣으면 된다
    '''
    if type(sentence) == str:
        p = re.compile(fr'''
        (
        ((http|https)\:\/\/)?            # http가 존재하거나 존재하지 않거나
        [a-zA-Z0-9\.\-_]+\.              # 웹주소 .kr 이전 까지만 선택
        ([a-zA-Z]){2,6}                  # .kr, .org 등을 모두 선택
        ([a-zA-Z0-9\.\&\/\?\:@\-_=#])*   # 파라미터 선택 / 이메일 선택
        )|
        (
        \([^)]*\)                        # () 안에 내용까지 선택 (반각)    
        )|
        (
        \<[^>]*\>                        # <> 안에 내용까지 선택 (반각)
        )|
        (
        \[[^\]]*\]                       # [] 안에 내용까지 선택 (반각)
        )|
        (
        --[^-]*--                        # -- 내용 -- 내용까지 선택 (반각)
        )|
        (
        \u00ab[^\u00bb]*\u00bb           # <> 안에 내용까지 선택 (반각)
        )|
        (
        \uff1c[^\uff1e]*\uff1e           # < > 안에 내용까지 선택 (전각)
        )|
        (
        \ufe64[^\ufe65]*\ufe65           # << >> 안에 내용까지 선택 (전각)
        )|
        (
        \uff08[^\uff09]*\uff09           # () 안에 내용까지 선택 (전각)    
        )|
        (
        \ufe59[^\ufe5a]*\ufe5a            # () 안에 내용까지 선택 (전각) 
        )|
        (
        \uff3b[^\uff3d]*\uff3d           # [] 안에 내용까지 선택 (전각)
        )|
        (                                
        [^ {unicode}|  # 사용할 언어

        0-9|0-9 /|0-9/|0-9 :|0-9:|       # 숫자와 숫자 뒤 /와 :는 선택에서 제외

        ·|.|,|!|?|"|"|'|⸢|⸥|。|          # 포함할 특수문자 (반각)
                                         # 포함할 특수문자 (전각) :
        \uff0e|\ufe52|\uff0c|\ufe51|     # 마침표, 콤마, 느낌표,
        \ufe50|\ufe57|\uff01|\ufe15|     # 물음표, 큰따옴표,
        \uff1f|\ufe56|\ufe16|\uff02|     # 어퍼스트로피, 꺽새 (전각)
        \uff07|\ufe41|\ufe42|\ufe43|\ufe44]
        )                               
        ''', re.VERBOSE)
        result = p.sub('', sentence)
    else:
        result = ''
    return result


def data_splitter(data, num_split):
    data_index = []
    diff = len(data) // num_split
    for i in range(0, len(data) + 1, diff):
        start = i
        end = i + diff
        data_index.append([data[start:end], start])

        if len(data_index) == num_split - 1:
            data_index.append([data[end : len(data) + 1], end])
            return data_index


def read_data(dirname, filename):
    filepath = os.path.join(dirname, filename)
    with open(filepath, 'rb') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.decode().strip()
            data.append(line)
    
    return data


# 데이터는 전체 데이터를 추출하고, 인덱스는 해당 언어에 맞는 데이터의 인덱스만 추출
def get_src(data_index):
    global SRC_UNICODE
    global SRC_LANGCODE

    data = data_index[0]
    num_index = data_index[-1]

    p = re.compile(fr'''([^ {SRC_UNICODE}])''', re.VERBOSE)
    index = []
    for line in data:
        if p.sub(" ", line).isspace() is not True:  # 해당 유니코드의 범위안에 있는 글자가 하나라도 있을때
            try:
                if lang_detect.language_detection_langdetect(line) == SRC_LANGCODE:
                    index.append(num_index)
                    num_index += 1
                else:
                    num_index += 1
                    pass
            except:
                # print(f'line:{line}')
                index.append(num_index)
                num_index += 1
        else:
            num_index += 1
    
    return index


# 데이터는 전체 데이터를 추출하고, 인덱스는 해당 언어에 맞는 데이터의 인덱스만 추출
def get_tgl(data_index):
    
    data = data_index[0]
    num_index = data_index[-1]

    p = re.compile(fr'''([^ {tl}])''', re.VERBOSE)
    index = []
    for line in data:
        if p.sub(" ", line).isspace() is not True:  # 해당 유니코드의 범위안에 있는 글자가 하나라도 있을때
            try:
                if lang_detect.language_detection_langdetect(line) == "tl":
                    index.append(num_index)
                    num_index += 1
                else:
                    num_index += 1
                    pass
            except:
                # print(f'line:{line}')
                index.append(num_index)
                num_index += 1
        else:
            num_index += 1
    
    return index

# 데이터길이가 너무 짧아서 멀티로 돌릴 필요가 없을 때
def get_data(unicode, data, langcode):
    '''
    데이터 전체를 읽어오고, 인덱스는 해당 언어에 맞는 데이터의 인덱스만 추출
    전체 데이터 중 겹치는 인덱스의 데이터만 추출하여 사용할 예정
    '''
    p = re.compile(fr'''([^ {unicode}])''', re.VERBOSE)

    i = 0
    index = []
    for line in tqdm(data):
        if p.sub(" ", line).isspace() is not True:  # 해당 유니코드의 범위안에 있는 글자가 하나라도 있을때
            try:
                if lang_detect.language_detection_langdetect(line) == langcode:
                    index.append(i)
                    i += 1
                else:
                    i += 1
                    pass
            except:
                index.append(i)
                i += 1
        else:
            i += 1
    
    return index


def parallel_work(src_data, tgl_data, src_func, tgl_func, num_cores):
    src_index = data_splitter(src_data, num_cores)
    tgl_index = data_splitter(tgl_data, num_cores)

    src_result = sorted(np.concatenate(parmap.map(src_func, src_index, pm_pbar=True, pm_processes=num_cores)).tolist())
    tgl_result = sorted(np.concatenate(parmap.map(tgl_func, tgl_index, pm_pbar=True, pm_processes=num_cores)).tolist())

    # 공통 인덱스 추출
    lang_index = set(src_result) & set(tgl_result)
    lang_index = list(lang_index)

    return lang_index


def main():
    global SRC_UNICODE
    global SRC_LANGCODE

    src_lang = input("put the src_lang (ex. kor) : ")
    dirname = f"/Users/ujinne/python/mtdata/data/{src_lang}-tgl/train-parts"
    print(f"dirname : {dirname}")

    file_list = os.listdir(dirname)
    src_list = sorted([f for f in file_list if f.endswith(f".{src_lang}")])
    tgl_list = sorted([f for f in file_list if f.endswith(".tgl")])
    # print(f'src_list : {src_list}')
    # print(f'tgl_list : {tgl_list}')

    data = []
    for i in range(len(tgl_list)):
        print(f'src_list : {src_list[i]}')
        print(f'tgl_list : {tgl_list[i]}')

        src_data = read_data(dirname, src_list[i])
        tgl_data = read_data(dirname, tgl_list[i])
        print(f'src_data length : {len(src_data)}')
        print(f'tgl_data length : {len(tgl_data)}')
        
        if len(tgl_data) >= 8:
            lang_index = parallel_work(src_data, tgl_data, get_src, get_tgl, 8)
            print(f'preprocessed data length : {len(lang_index)}')
        else:
            src_index = get_data(SRC_UNICODE, src_data, SRC_LANGCODE)
            tgl_index = get_data(tl, tgl_data, "tl")
            lang_index = set(src_index) & set(tgl_index)
            lang_index = list(lang_index)
            print(f'preprocessed data length : {len(lang_index)}')

        # 인덱스에 맞춰 공통되는 데이터만 추출 후 포맷에 맞게 입력
        # src_data = [src_data[i] for i in lang_index]
        # tgl_data = [tgl_data[i] for i in lang_index]
        if len(lang_index) != 0:
            for num in tqdm(lang_index):
                content = {
                            "src_lang": SRC_LANGCODE,
                            "src_text_raw": src_data[num],
                            "src_text": clean_lng(src_data[num], SRC_UNICODE),
                            "tgt_lang": "tl",
                            "tgt_text_raw": tgl_data[num],
                            "tgt_text": clean_lng(tgl_data[num], tl),
                            "origin": "mtdata",
                            "domain": " "
                            }
                
                data.append(content)
        else:
            pass

        print("--------------------F I N I S H--------------------")
        
    save_path = "/Users/ujinne/python/mtdata/result/"
    file_name = f'{SRC_LANGCODE}_tl.json'
    save_json(save_path+file_name, data)

    print("--------------------Saved the file--------------------")


if __name__ == "__main__":
    lang_detect = LangDetector()

    main()

    # save_path = "/Users/ujinne/python/mtdata/result/"
    # ## file_name 바꿔주기!!
    # file_name = f'{src_langcode}_tl.json'
    # save_json(save_path+file_name, data)
            
