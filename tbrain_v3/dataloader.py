import json
import re
import os
from tqdm import tqdm
from pathlib import Path
from opencc import OpenCC
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from tbrain_v3.settings import settings


class DataLoader:
    def __init__(self):
        self.reference_path = settings.source_path
        self.question_path = settings.question_path

        self.opencc = OpenCC("s2t")
        if settings.embedding_model == "bge-m3":
            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

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

        corpus_name = "corpus_v3"
        if settings.clean_text:
            corpus_name += "_cleaned"
        dataset_json_path = Path(self.reference_path) / f"{corpus_name}.json"
        dataset_embedding_folder = (
            Path(self.reference_path) / f"{corpus_name}_embedding"
        )
        dataset_embedding_folder.mkdir(parents=True, exist_ok=True)

        question_json_name = "questions_v3"
        if settings.clean_text:
            question_json_name += "_cleaned"
        question_json_path = (
            Path(self.question_path).parent / f"{question_json_name}.json"
        )
        question_embedding_folder = (
            Path(self.question_path).parent / f"{question_json_name}_embedding"
        )
        question_embedding_folder.mkdir(parents=True, exist_ok=True)

        if dataset_json_path.exists():
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
                    tokens = self.embedding_model.tokenizer.encode(text)
                    split_texts = []
                    for i in range(0, len(tokens), settings.stride):
                        split_texts.append(
                            self.embedding_model.tokenizer.decode(
                                tokens[i : i + settings.max_tokens]
                            )
                        )
                        if i + settings.max_tokens > len(tokens):
                            break
                    embedding_path = (
                        dataset_embedding_folder / f"{category}_{corpus_id}.npy"
                    )
                    if not embedding_path.exists():
                        embeddings = self.embedding_model.encode(
                            split_texts, batch_size=4
                        )["dense_vecs"]
                        np.save(embedding_path, embeddings)
                    dataset[category][str(corpus_id)] = str(embedding_path)

                pbar.update(1)

        pbar = tqdm(total=len(questions))
        for question in questions:
            query = question["query"]
            query = self._remove_stopwords(query)
            tokens = self.embedding_model.tokenizer.encode(query)
            if len(tokens) > 8192:
                print(f"query length: {len(tokens)} exceed 8192, truncate to 8192")
                query = self.embedding_model.tokenizer.decode(tokens[:8192])
            embedding_path = question_embedding_folder / f"{question['qid']}.npy"
            if not embedding_path.exists():
                embedding = self.embedding_model.encode(query)["dense_vecs"]
                np.save(embedding_path, embedding)
            question["query_embedding"] = str(embedding_path)
            pbar.update(1)

        with open(dataset_json_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        with open(question_json_path, "w") as f:
            json.dump(questions, f, ensure_ascii=False, indent=4)

        return questions, dataset
