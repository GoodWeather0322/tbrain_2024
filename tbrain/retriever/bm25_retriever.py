from rank_bm25 import BM25Okapi
import jieba
import json


class BM25Retriever:
    def __init__(self):
        pass

    def retrieve(self, query: str, source: list[int], corpus_dict: dict):
        filtered_corpus = [corpus_dict[int(file)] for file in source]

        # [TODO] 可自行替換其他檢索方式，以提升效能

        tokenized_corpus = [
            list(jieba.cut_for_search(doc)) for doc in filtered_corpus
        ]  # 將每篇文檔進行分詞
        bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = list(jieba.cut_for_search(query))  # 將查詢語句進行分詞
        ans = bm25.get_top_n(
            tokenized_query, list(filtered_corpus), n=1
        )  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        a = ans[0]
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for key, value in corpus_dict.items() if value == a]
        return res[0]  # 回傳檔案名

    def retrieve_all(self, question_path, corpus_dict):
        with open(question_path, "rb") as f:
            qs_ref = json.load(f)  # 讀取問題檔案

        answer_dict = {"answers": []}  # 初始化字典

        for q_dict in qs_ref["questions"]:
            if q_dict["category"] == "finance":
                # 進行檢索
                retrieved = self.retrieve(
                    q_dict["query"], q_dict["source"], corpus_dict["finance"]
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "insurance":
                retrieved = self.retrieve(
                    q_dict["query"], q_dict["source"], corpus_dict["insurance"]
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "faq":
                retrieved = self.retrieve(
                    q_dict["query"], q_dict["source"], corpus_dict["faq"]
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

        return answer_dict
