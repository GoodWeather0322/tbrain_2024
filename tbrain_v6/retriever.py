from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker

from tbrain_v6.settings import settings


class Retriever:
    def __init__(self):
        pass

    def _lcs_length(self, query, document):
        # 將查詢和文檔分割為單詞列表
        if isinstance(query, list):
            query_words = query
        else:
            query_words = query.split()

        if isinstance(document, list):
            document_words = document
        else:
            document_words = document.split()

        # 初始化動態規劃表格，大小為 (len(query_words) + 1) x (len(document_words) + 1)
        dp = [[0] * (len(document_words) + 1) for _ in range(len(query_words) + 1)]

        # 填充 DP 表格
        for i in range(1, len(query_words) + 1):
            for j in range(1, len(document_words) + 1):
                if query_words[i - 1] == document_words[j - 1]:  # 單詞匹配
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # LCS 長度位於表格的右下角，除以最短的長度做normalize
        return dp[len(query_words)][len(document_words)] / min(
            len(query_words), len(document_words)
        )

    def _fusion_retrieve(
        self, query_bm25, query_embedding_file, source, corpus_bm25, corpus_embedding
    ):
        # LCS Score
        lcs_scores = []
        for file in source:
            source_text = corpus_bm25[str(file)]
            lcs_length = self._lcs_length(query_bm25, source_text)
            lcs_scores.append(lcs_length)

        lcs_scores = np.array(lcs_scores)

        # BM25 Score
        filtered_corpus = [corpus_bm25[str(file)] for file in source]
        tokenized_corpus = filtered_corpus
        bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
        tokenized_query = query_bm25  # 將查詢語句進行分詞
        bm25_scores = bm25.get_scores(
            tokenized_query
        )  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
        if settings.normalize == "minmax":
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
                np.max(bm25_scores) - np.min(bm25_scores)
            )
        elif settings.normalize == "zscore":
            bm25_scores = (bm25_scores - np.mean(bm25_scores)) / np.std(bm25_scores)

        # Embedding Score
        lcs_scores_match_length = []
        bm25_scores_match_length = []
        corpus_embeddings = []
        corpus_ids = []
        for i, file in enumerate(source):
            embedding_file = corpus_embedding[str(file)]
            embeddings = np.load(embedding_file)
            corpus_embeddings.append(embeddings)
            corpus_ids += [file] * embeddings.shape[0]
            bm25_scores_match_length += [bm25_scores[i]] * embeddings.shape[0]
            lcs_scores_match_length += [lcs_scores[i]] * embeddings.shape[0]
        corpus_embeddings = np.vstack(corpus_embeddings)
        assert corpus_embeddings.shape[0] == len(corpus_ids)
        query_embedding = np.load(query_embedding_file)
        # 計算餘弦相似度
        cosine_similarity = np.dot(query_embedding, corpus_embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(corpus_embeddings, axis=1)
        )

        # 融合
        assert (
            len(bm25_scores_match_length)
            == len(cosine_similarity)
            == len(lcs_scores_match_length)
        )
        fusion_scores = []
        for i, (lcs_score, bm25_score, cosine_score) in enumerate(
            zip(lcs_scores_match_length, bm25_scores_match_length, cosine_similarity)
        ):
            score = (
                settings.beta * lcs_score
                + settings.alpha * bm25_score
                + (1 - settings.alpha - settings.beta) * cosine_score
            )
            fusion_scores.append(score)
        max_index = np.argmax(fusion_scores)
        max_id = corpus_ids[max_index]
        return max_id

    def fusion_retrieve(
        self, questions_bm25, dataset_bm25, questions_embedding, dataset_embedding
    ):

        answer_dict_name = "answer_v6"
        if settings.clean_text:
            answer_dict_name += "_clean"
        answer_dict_name += f"_shallow_fusion_{settings.tokenizer}_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}_alpha_{settings.alpha}_beta_{settings.beta}.json"
        answer_dict_path = settings.output_dir / answer_dict_name

        answer_dict = {"answers": []}  # 初始化字典
        pbar = tqdm(total=len(questions_bm25))

        for bm25_dict, embedding_dict in zip(questions_bm25, questions_embedding):
            if bm25_dict["category"] == "finance":
                # 進行檢索
                retrieved = self._fusion_retrieve(
                    bm25_dict["query_ws"],
                    embedding_dict["query_embedding"],
                    bm25_dict["source"],
                    dataset_bm25["finance"],
                    dataset_embedding["finance"],
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": bm25_dict["qid"], "retrieve": retrieved}
                )

            elif bm25_dict["category"] == "insurance":
                retrieved = self._fusion_retrieve(
                    bm25_dict["query_ws"],
                    embedding_dict["query_embedding"],
                    bm25_dict["source"],
                    dataset_bm25["insurance"],
                    dataset_embedding["insurance"],
                )
                answer_dict["answers"].append(
                    {"qid": bm25_dict["qid"], "retrieve": retrieved}
                )

            elif bm25_dict["category"] == "faq":
                retrieved = self._fusion_retrieve(
                    bm25_dict["query_ws"],
                    embedding_dict["query_embedding"],
                    bm25_dict["source"],
                    dataset_bm25["faq"],
                    dataset_embedding["faq"],
                )
                answer_dict["answers"].append(
                    {"qid": bm25_dict["qid"], "retrieve": retrieved}
                )

            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

            pbar.update(1)

        with open(answer_dict_path, "w", encoding="utf8") as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)

        return answer_dict, answer_dict_path

    def retrieve(
        self, questions_bm25, dataset_bm25, questions_embedding, dataset_embedding
    ):
        return self.fusion_retrieve(
            questions_bm25, dataset_bm25, questions_embedding, dataset_embedding
        )
