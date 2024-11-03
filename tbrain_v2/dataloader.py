import json
import re
import os
from tqdm import tqdm
from opencc import OpenCC
import jieba
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

from tbrain_v2.settings import settings


class DataLoader:
    def __init__(self):
        self.reference_path = settings.source_path
        self.question_path = settings.question_path

        self.opencc = OpenCC("s2t")

    def load_dataset(self):
        # 先找出跟問題相關的reference，不要做多餘運算
        related_reference = {
            "insurance": [],
            "finance": [],
            "faq": [],
        }
        with open(self.question_path, "r") as f:
            data = json.load(f)
        questions = data["questions"]
        for question in questions:
            source = question["source"]
            category = question["category"]
            related_reference[category].extend(source)
        for category, source in related_reference.items():
            related_reference[category] = sorted(list(set(source)))

        return questions, related_reference

    def _opencc_convert(self, text: str):
        text = self.opencc.convert(text)
        return text

    def _ckip_tokenize(self, text_list: list[str]):  # list[str] -> list[str]
        ws_list = self.ckip_ws(text_list, show_progress=False)
        ws_list = [word for sentence in ws_list for word in sentence]
        return ws_list

    def _jieba_tokenize(self, text_list: list[str]):  # list[str] -> list[str]
        texts = " ".join(text_list)
        texts = texts.replace(" ", "")
        ws_list = list(jieba.cut_for_search(texts))
        return ws_list

    def _remove_stopwords(self, text: str):
        text = re.sub(r"\*\*page \d+\*\*", "", text)
        text = re.sub(r"\*\*question \d+\*\*", "", text)
        text = re.sub(r"\*\*answer \d+\*\*", "", text)
        # Step 1: 去除網址和 EMAIL
        text = re.sub(r"http\S+|www\S+|https\S+|[\w\.-]+@[\w\.-]+", "", text)

        text = re.sub(r"【[A-Za-z0-9]+】", "", text)

        # Step 6: 去除 "第 X 頁，共 Y 頁" 格式
        text = re.sub(r"第 \d+ 頁，共 \d+ 頁", "", text)

        # Step 7: 去除 "X/Y" 或 "X / Y" 格式
        text = re.sub(r"\b\d+ ?/ ?\d+\b", "", text)

        # Step 8: 去除 "~X~" 格式
        text = re.sub(r"~\d+~", "", text)

        # Step 9: 去除 "（接次頁）" 和 "（承前頁）"
        text = re.sub(r"（接次頁）|（承前頁）", "", text)

        # Step 10: 去除 "- X -" 格式
        text = re.sub(r"- \d+ -", "", text)

        # Step 2: 去除無意義數字（可以依需求調整，如果想保留某些數字格式）
        text = re.sub(r"\b\d+\b", "", text)

        # Step 3: 去除標點符號
        text = re.sub(r"[^\w\s]", "", text)

        # 去除多餘的空格
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_dataset(self, questions: list, related_corpus_ids: dict):

        corpus_name = "corpus_v2"
        if settings.clean_text:
            corpus_name += "_cleaned"
        if settings.tokenizer == "ckip":
            corpus_name += "_ckip"
        elif settings.tokenizer == "jieba":
            corpus_name += "_jieba"
        dataset_json_path = os.path.join(self.reference_path, f"{corpus_name}.json")

        question_json_name = "questions_v2"
        if settings.clean_text:
            question_json_name += "_cleaned"
        if settings.tokenizer == "ckip":
            question_json_name += "_ckip"
        elif settings.tokenizer == "jieba":
            question_json_name += "_jieba"
        question_json_path = os.path.join(
            os.path.dirname(self.question_path), f"{question_json_name}.json"
        )

        if os.path.exists(dataset_json_path):
            print(f"load dataset from {dataset_json_path}")
            with open(dataset_json_path, "r") as f:
                dataset = json.load(f)

            with open(question_json_path, "r") as f:
                questions = json.load(f)

            return questions, dataset

        self.ckip_ws = CkipWordSegmenter("bert-base")
        dataset = {
            "insurance": {},
            "finance": {},
            "faq": {},
        }

        total = 0
        for category, corpus_ids in related_corpus_ids.items():
            total += len(corpus_ids)

        pbar = tqdm(total=total)
        for category, corpus_ids in related_corpus_ids.items():
            for corpus_id in corpus_ids:
                with open(
                    f"{self.reference_path}/{category}/{corpus_id}.txt", "r"
                ) as f:
                    text = f.read()
                    text = self._opencc_convert(text)
                    texts = [text]
                    texts = [text.replace(" ", "") for text in texts]
                    if settings.clean_text:
                        texts = [
                            self._remove_stopwords(text) for text in texts
                        ]  # list[str] -> list[str]
                    if settings.tokenizer == "ckip":
                        ws_list = self._ckip_tokenize(texts)
                    elif settings.tokenizer == "jieba":
                        ws_list = self._jieba_tokenize(texts)
                    dataset[category][str(corpus_id)] = ws_list

                pbar.update(1)

        pbar = tqdm(total=len(questions))
        for question in questions:
            query = question["query"]
            query = self._remove_stopwords(query)
            if settings.tokenizer == "ckip":
                query_ws = self._ckip_tokenize([query])
            elif settings.tokenizer == "jieba":
                query_ws = self._jieba_tokenize([query])
            question["query_ws"] = query_ws
            pbar.update(1)

        with open(dataset_json_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        with open(question_json_path, "w") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        return questions, dataset
