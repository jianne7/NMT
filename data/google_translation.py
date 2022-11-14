import pandas as pd
import six
from google.cloud import translate_v2 as translate

# random.seed(777)

# def implicit():
#     from google.cloud import storage

#     # If you don't specify credentials when constructing the client, the
#     # client library will look for credentials in the environment.
#     storage_client = storage.Client()

#     # Make an authenticated API request
#     buckets = list(storage_client.list_buckets())
#     print(buckets)


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    # import six
    # from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    try:
        result = translate_client.translate(text, target_language=target)
        translated = result["translatedText"]
        
    except:
        translated = " "
    
    print(f"src_text:{text}")
    print(f"tgt_lang:{target}")
    print(f"translated:{translated}")

    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

    return translated


if __name__ == "__main__":
    valid_path = "/Users/ujinne/python/Lamp/Data/neologism.xlsx"
    # Read CSV
    # data = pd.read_csv(valid_path, encoding='utf-8-sig')
    # Read Excel
    data = pd.read_excel(valid_path)
    # data = data[:10]
    # print(len(data))

    tgt_data = []
    for i in range(len(data)):
        print(f"index_num:{i}")
        src_text = data["단어"][i]
        # print(src_text)
        tgt_lang = "en"
        translated = translate_text(tgt_lang, src_text)
        tgt_data.append(translated)

    data["Google"] = tgt_data
    data.to_csv("/Users/ujinne/python/Lamp/Data/neologism.csv", index=False, encoding="utf-8-sig")
