import os, sys, requests, tqdm, time
import fasttext
import langid
import langdetect


class LangDetector:
    def __init__(self):
        self._fasttext_lang_id = None

    def http_get(self, url, path):
        """
        Downloads a URL to a given path on disc
        """
        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            print(
                "Exception when trying to download {}. Response {}".format(
                    url, req.status_code
                ),
                file=sys.stderr,
            )
            req.raise_for_status()
            return

        download_filepath = path + "_part"
        with open(download_filepath, "wb") as file_binary:
            content_length = req.headers.get("Content-Length")
            total = int(content_length) if content_length is not None else None
            progress = tqdm.tqdm(unit="B", total=total, unit_scale=True)
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    file_binary.write(chunk)

        os.rename(download_filepath, path)
        progress.close()

    def language_detection_fasttext(self, text: str) -> str:
        if self._fasttext_lang_id is None:
            fasttext.FastText.eprint = (
                lambda x: None
            )  # Silence useless warning: https://github.com/facebookresearch/fastText/issues/1067
            model_path = "/Users/ujinne/Downloads/lid.176.ftz"
            if not os.path.exists(model_path):
                self.http_get(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
                    model_path,
                )
            self._fasttext_lang_id = fasttext.load_model(model_path)

        start = time.time()

        result = self._fasttext_lang_id.predict(
            text.lower().replace("\r\n", " ").replace("\n", " ").strip()
        )[0][0].split("__")[-1]

        end = time.time()

        return (result, end - start)

    def language_detection_langid(self, text: str) -> str:
        return langid.classify(
            text.lower().replace("\r\n", " ").replace("\n", " ").strip()
        )[0]

    def language_detection_langdetect(self, text: str) -> str:
        return langdetect.detect(
            text.lower().replace("\r\n", " ").replace("\n", " ").strip()
        ).split("-")[0]


# detector = LangDetector()

# # 반갑다
# gan_text = "很高兴认识"  # 간체
# bun_text = "很高興認識"  # 번체

# for lang, text in zip(("GAN", "BUN"), (gan_text, bun_text)):
#     print(f"LANG: {lang}")

#     result = detector.language_detection_fasttext(text)
#     print(f"FastText: {result[0]}")
#     print(result[1], end="\n\n")

#     start = time.time()
#     print(f"LangID: {detector.language_detection_langid(text)}")
#     end = time.time()
#     print(end - start, end="\n\n")

#     start = time.time()
#     print(f"LangDetect: {detector.language_detection_langdetect(text)}")
#     end = time.time()
#     print(end - start, end="\n\n")