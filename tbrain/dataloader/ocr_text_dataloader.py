import os
import json
from tqdm import tqdm
from opencc import OpenCC
from tbrain.config.settings import settings

opencc = OpenCC("s2t")


class OcrTextDataLoader:
    def __init__(self, source_path: str):
        self.source_path = source_path
        if settings.tokenizer == "jieba":
            self._load_data = self._load_data_jieba
        elif settings.tokenizer == "ckip":
            self._load_data = self._load_data_ckip

    def _load_data_jieba(self, source_path):
        masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
        corpus_dict = {}
        for file in tqdm(masked_file_ls):
            with open(os.path.join(source_path, file), "r") as f:
                texts = f.read()
                converted_texts = opencc.convert(texts)
                converted_texts_nospace = converted_texts.replace(" ", "")
                corpus_dict[int(file.replace(".txt", ""))] = converted_texts_nospace

        return corpus_dict

    def _load_data_ckip(self, source_path):
        masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
        corpus_dict = {}
        for file in tqdm(masked_file_ls):
            with open(os.path.join(source_path, file), "r") as f:
                texts = f.read()
                texts = texts.replace("\n", " ")
                corpus_dict[int(file.replace(".txt", ""))] = texts

        return corpus_dict

    def load_data(self):
        data_json_path = os.path.join(self.source_path, "data.json")

        corpus_dict_insurance = self._load_data(
            os.path.join(self.source_path, "insurance")
        )
        corpus_dict_finance = self._load_data(os.path.join(self.source_path, "finance"))
        corpus_dict_faq = self._load_data(os.path.join(self.source_path, "faq"))

        corpus_dict = {
            "insurance": corpus_dict_insurance,
            "finance": corpus_dict_finance,
            "faq": corpus_dict_faq,
        }
        with open(data_json_path, "w") as f:
            json.dump(corpus_dict, f, ensure_ascii=False, indent=4)

        return {
            "insurance": corpus_dict_insurance,
            "finance": corpus_dict_finance,
            "faq": corpus_dict_faq,
        }
