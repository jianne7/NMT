import re
import os
import time
import requests
import logging
from urllib3.exceptions import NewConnectionError, MaxRetryError, ConnectTimeoutError

import json
import random
import numpy as np
import parmap
# from tqdm import tqdm

from pymongo import MongoClient
from multiprocessing import Pool

# file_name
FILE_NAME = 'relationship.json'


def load_json(file_path):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        data = json.load(f, strict=False)

    return data


def save_json(path: str, json_inputs: dict):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(json.dumps(json_inputs, indent='\t', ensure_ascii=False))


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


# 동북 아시아
ko = "\u3131-\u3163|\uac00-\ud7af|\u0041-\u007a"  # ko + en
zh = "\u4e00-\u62ff|\u6300-\u77ff|\u7800-\u8cff|\u8d00-\u9fff|\u0041-\u007a"  # zh_cn + en
ja = "\u3041-\u3096|\u30a0-\u30ff|\u3400-\u4db5|\u4e00-\u9fcb|\uf900-\ufa6a|\u0041-\u007a"  # ja + en

# 러시아
ru = "\u0410-\u044f"  # ru

# 유럽 / 미국 (vi, de, pt, fr, id, es, it, en)
for country in "vi,de,pt,fr,id,es,it,en".split(","):
    globals()[
        f"{country}"
    ] = "\u0041-\u007a|\u00c0-\u0178|\u1e00-\u1eff|\u0180-\u024f|\u1e00—\u1eff|\u0027"

# 인도
hi = "\u0900-\u097f|\ua8e0-\ua8ff"  # hi (드 파나 가리어)

# 아랍
ar = "\u0627-\u064a"  # ar


def clean_lng(sentence, unicode):
    """
    문장 전처리 함수
    문장과 해당 언어의 유니코드를 문자열로 넣으면 된다
    """
    if type(sentence) == str:
        p = re.compile(
            fr"""
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
        """,
            re.VERBOSE,
        )
        result = p.sub("", sentence)
    else:
        result = ""
    return result


def get_translated(src_text, tgt_lang, logger):
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
    client_id = "uxb026vbjn"
    client_secret = "OQvVVjw61mX4p1ZSYzWuW4Kh0ridOKrKgoGn9GL0"
    header = {"X-NCP-APIGW-API-KEY-ID" : client_id,
            "X-NCP-APIGW-API-KEY" : client_secret}

    data = {'text' : src_text,
            'source' : 'ko',
            'target' : tgt_lang}

    while True:
        try:
            response = requests.post(url, headers=header, data=data, timeout=10)
            break
        except requests.Timeout:
            logger.error('-------Timeout ERROR-------')
        except requests.exceptions.ConnectionError:
            logger.error('-------Connection ERROR-------')
        except (NewConnectionError, MaxRetryError, ConnectTimeoutError):
            logger.exception("------Exception ERROR------")
        except:
            logger.error('------Unexpected Error------')
        time.sleep(2)

    rescode = response.status_code

    if rescode == 200:
        translated = response.json()['message']['result']['translatedText']
        # print(f"translated : {translated}")

    else:
        print("Error Code:" + str(rescode))
        logger.info(f"Error Code: {rescode}")
        translated = " "

    return translated


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


def work_func(data_index):
    # db 연결
    global FILE_NAME
    db_client = MongoClient("mongodb://localhost:27017/")
    db = db_client["papago"]
    col = db[FILE_NAME[:-5]]

    make_directory('papago_log')
    logger = get_logger(name='papago', file_path=os.path.join('papago_log', FILE_NAME[:-5] + ".log"), stream=True)

    data = data_index[0]
    num_index = data_index[-1]

    #num_index = index
    tgt_lst = ["en", "zh-CN", "es", "fr", "ru", "id", "ja", "vi", "de", "it"]
    for item in data:
        #print(f'num_index : {num_index}')
        src_text = item['src_text']
        #print(f'src_text : {src_text}')
        tgt_lang = random.choice(tgt_lst)
        translated = get_translated(src_text, tgt_lang, logger)

        logger.info(f'num_index: {num_index}')
        logger.info(f'src_text: {src_text}')
        logger.info(f'translated: {translated}')
        #print(f"translated : {translated}")
        item['tgt_lang'] = tgt_lang[:2]
        item['tgt_text_raw'] = translated
        
        # text cleaning
        if item["tgt_lang"] == "zh":
            lang_unicode = zh
        elif item["tgt_lang"] == "ja":
            lang_unicode = ja
        elif item["tgt_lang"] == "ru":
            lang_unicode = ru
        else:
            lang_unicode = en

        tgt_text = clean_lng(translated, lang_unicode)
        item["tgt_text"] = tgt_text

        # db 저장
        db_post = {"num_index" : num_index,
                   "src_lang" : "ko",
                   "src_text" : src_text,
                   "tgt_lang" : tgt_lang[:2],
                   "tgt_text_raw" : translated,
                   "tgt_text" : tgt_text}
        col.insert_one(db_post)

        num_index += 1
    
    return data


# TEST!!!!!!!!!!!!!!!!!!!!!!
def test_func(data_index):
    # global FILE_NAME
    db_client = MongoClient("mongodb://localhost:27017/")
    db = db_client["papago"]
    col = db['test']

    data = data_index[0]
    # print(f"data:{data}")
    index = data_index[-1]
    # print(f"index:{index}")
    for item in data:
        # print(f"index:{index}")
        item["index"] = index
        item["tgt_text"] = "This is the TEST"

        # db 저장
        result = {
            "num_index": index,
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"],
        }
        col.insert_one(result)

        index += 1
        # print("-" * 100)
        # break
    return data


def parallel_work(data, func, num_cores):
    data_index = data_splitter(data, num_cores)
    # pool = Pool(num_cores)
    # result = np.concatenate(pool.map(func, data_index)).tolist()
    
    # pool.close()
    # pool.join()  # One must call close() or terminate() before using join().

    result = np.concatenate(parmap.map(func, data_index, pm_pbar=True, pm_processes=num_cores)).tolist()

    return result


def main():
    global FILE_NAME

    raw_dir = '/Users/ujinne/python/Lamp/Data/Parsed_Data/032_Korean SNS/'
    data = load_json(os.path.join(raw_dir, FILE_NAME))

    result = parallel_work(data, work_func, 8)
    # result = parallel_work(data, test_func, 8)

    save_dir = '/Users/ujinne/python/Lamp/Data/Parsed_Data/032_Korean SNS/result'
    save_json(os.path.join(save_dir, FILE_NAME), result)


if __name__ == "__main__":
    main()

