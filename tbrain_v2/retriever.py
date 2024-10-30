import os
import json
from rank_bm25 import BM25Okapi

from tbrain_v2.settings import settings


class Retriever:
    def __init__(self):
        pass

    def _bm25_retrieve(self, query: list[str], source: list[int], corpus_dict: dict):
        filtered_corpus = [corpus_dict[str(file)] for file in source]
        tokenized_corpus = filtered_corpus
        with open("tokenized_corpus_v2.txt", "w", encoding="utf8") as f:
            for doc in tokenized_corpus:
                f.write(str(doc) + "\n")
        bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = query  # 將查詢語句進行分詞
        with open("tokenized_query_v2.txt", "w", encoding="utf8") as f:
            f.write(str(tokenized_query) + "\n")
        ans = bm25.get_top_n(
            tokenized_query, list(filtered_corpus), n=1
        )  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        a = ans[0]
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for key, value in corpus_dict.items() if value == a]
        return int(res[0])  # 回傳檔案名

    def bm25_retrieve(self, questions, dataset):

        answer_dict_name = "answer_v2"
        if settings.retriever == "bm25":
            answer_dict_name += "_bm25"
        if settings.clean_text:
            answer_dict_name += "_clean"
        if settings.tokenizer == "ckip":
            answer_dict_name += "_ckip"
        elif settings.tokenizer == "jieba":
            answer_dict_name += "_jieba"

        answer_dict_name += ".json"
        answer_dict_path = os.path.join(settings.output_dir, answer_dict_name)

        answer_dict = {"answers": []}  # 初始化字典

        for q_dict in questions:
            if q_dict["category"] == "finance":
                # 進行檢索
                retrieved = self._bm25_retrieve(
                    q_dict["query_ws"], q_dict["source"], dataset["finance"]
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "insurance":
                retrieved = self._bm25_retrieve(
                    q_dict["query_ws"], q_dict["source"], dataset["insurance"]
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "faq":
                retrieved = self._bm25_retrieve(
                    q_dict["query_ws"], q_dict["source"], dataset["faq"]
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        with open(answer_dict_path, "w", encoding="utf8") as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)

        return answer_dict

    def retrieve(self, questions, dataset):
        if settings.retriever == "bm25":
            return self.bm25_retrieve(questions, dataset)
