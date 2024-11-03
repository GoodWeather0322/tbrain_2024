import json
import re
import os
from tqdm import tqdm
from pathlib import Path
from opencc import OpenCC
import numpy as np
from FlagEmbedding import FlagReranker

from tbrain_v6.settings import settings


class DataLoader:
    def __init__(self):
        self.reference_path = settings.source_path
        self.question_path = settings.question_path

        self.opencc = OpenCC("s2t")
        if settings.reranker == "bge-reranker-v2-m3":
            self.rerank_model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def load_dataset(self):
        raise NotImplementedError

    def _opencc_convert(self, text: str):
        text = self.opencc.convert(text)
        return text

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
        raise NotImplementedError
