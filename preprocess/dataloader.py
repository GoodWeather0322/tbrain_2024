import json
import re
import os
from tqdm import tqdm
from pathlib import Path
from opencc import OpenCC
import numpy as np
from ckip_transformers.nlp import CkipWordSegmenter
from FlagEmbedding import BGEM3FlagModel

from settings import settings


class DataLoader:
    """資料預處理類。

    負責預處理文本數據，包括文本清理、分詞、embedding生成等功能。

    Attributes:
        reference_path (str): 參考文檔路徑
        question_path (str): 問題文檔路徑
        opencc (OpenCC): 繁簡轉換工具
        embedding_model (BGEM3FlagModel): embedding模型
    """

    def __init__(self):
        self.reference_path = settings.source_path
        self.question_path = settings.question_path

        self.opencc = OpenCC("s2t")

        if settings.embedding_model == "bge-m3":
            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        elif settings.embedding_model == "bge-large-zh-v1.5":
            self.embedding_model = BGEM3FlagModel(
                "BAAI/bge-large-zh-v1.5", use_fp16=True
            )

    def load_dataset(self):
        """載入資料集並找出相關的參考文檔。

        從問題文件中讀取資料，並按類別整理相關的參考文檔ID。

        Returns:
            tuple: (問題列表, 相關參考文檔字典)
                - questions (list): 問題數據列表
                - related_reference (dict): 按類別的參考文檔ID
        """
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
        """將文本從簡體轉換為繁體。

        Args:
            text (str): 輸入的簡體文本

        Returns:
            str: 轉換後的繁體文本
        """
        text = self.opencc.convert(text)
        return text

    def _remove_stopwords(self, text: str):
        """清理文本中的無用信息。

        移除標點符號、網址、E-mail、頁碼、
        問題編號、答案編號等文字。

        Args:
            text (str): 原始文本

        Returns:
            str: 清理後的文本
        """
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
        """預處理數據集。

        對問題和語料庫進行預處理，包括文本清理、
        生成embedding等，並將結果保存到文件。

        Args:
            questions (list): 問題列表
            related_corpus_ids (dict): 相關資料庫ID字典

        Returns:
            tuple: (預處理後的問題列表, 預處理後的數據集)
                - questions (list): 包含embedding路徑的問題列表
                - dataset (dict): 包含tokens和embedding路徑的資料集dict

        Notes:
            - 生成的embedding和預處理後的資料會被儲存到文件
            - 如果緩存文件存在，則直接從文件載入
        """
        corpus_name = "corpus_v10_competition"
        if settings.clean_text:
            corpus_name += "_cleaned"
        if settings.retriever == "sparse":
            corpus_name += "_sparse"
        dataset_json_path = (
            Path(self.reference_path)
            / f"{corpus_name}_embedding_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}.json"
        )
        dataset_embedding_folder = (
            Path(self.reference_path)
            / f"{corpus_name}_embedding_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}"
        )
        dataset_embedding_folder.mkdir(parents=True, exist_ok=True)

        question_json_name = "questions_v10_competition"
        if settings.clean_text:
            question_json_name += "_cleaned"
        if settings.retriever == "sparse":
            question_json_name += "_sparse"
        question_json_path = (
            Path(self.question_path).parent
            / f"{question_json_name}_embedding_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}.json"
        )
        question_embedding_folder = (
            Path(self.question_path).parent
            / f"{question_json_name}_embedding_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}"
        )
        question_embedding_folder.mkdir(parents=True, exist_ok=True)

        if dataset_json_path.exists():
            print(f"load dataset from {dataset_json_path}")
            with open(dataset_json_path, "r") as f:
                dataset = json.load(f)

            with open(question_json_path, "r") as f:
                questions = json.load(f)

            return questions, dataset

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
                    text = self._remove_stopwords(text)

                    tokens_list = self.embedding_model.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        max_length=settings.max_tokens,
                        stride=settings.stride,
                        return_overflowing_tokens=True,
                    )
                    split_texts = self.embedding_model.tokenizer.batch_decode(
                        tokens_list
                    )
                    embedding_path = (
                        dataset_embedding_folder / f"{category}_{corpus_id}.npy"
                    )
                    if not embedding_path.exists():
                        embeddings = self.embedding_model.encode(
                            split_texts,
                            return_dense=True,
                            return_sparse=True,
                            batch_size=4,
                        )
                        if settings.retriever == "sparse":
                            np.save(embedding_path, embeddings["lexical_weights"])
                        else:
                            np.save(embedding_path, embeddings["dense_vecs"])
                    dataset[category][str(corpus_id)] = {
                        "tokens": tokens_list,
                        "embedding_path": str(embedding_path),
                    }

                pbar.update(1)

        pbar = tqdm(total=len(questions))
        for question in questions:
            query = question["query"]
            query = self._remove_stopwords(query)
            tokens = self.embedding_model.tokenizer.encode(
                query,
                add_special_tokens=False,
                max_length=settings.max_tokens,
            )
            embedding_path = question_embedding_folder / f"{question['qid']}.npy"
            if not embedding_path.exists():
                if settings.retriever == "sparse":
                    query = [query]
                embedding = self.embedding_model.encode(
                    query,
                    return_dense=True,
                    return_sparse=True,
                )
                if settings.retriever == "sparse":
                    np.save(embedding_path, embedding["lexical_weights"])
                else:
                    np.save(embedding_path, embedding["dense_vecs"])
            question["query_embedding"] = str(embedding_path)
            pbar.update(1)

        with open(dataset_json_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        with open(question_json_path, "w") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        return questions, dataset
