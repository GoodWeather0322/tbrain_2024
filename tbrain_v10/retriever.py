from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker
from ckip_transformers.nlp import CkipWordSegmenter
from FlagEmbedding import BGEM3FlagModel

from tbrain_v10.settings import settings


class Retriever:
    def __init__(self):
        self.ckip_ws = CkipWordSegmenter("bert-base")

        if settings.embedding_model == "bge-m3":
            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        elif settings.embedding_model == "bge-large-zh-v1.5":
            self.embedding_model = BGEM3FlagModel(
                "BAAI/bge-large-zh-v1.5", use_fp16=True
            )

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
        self, category, qid, query, query_embedding_file, source, corpus
    ):
        score_dict_name = "topn_score_v10"
        if settings.clean_text:
            score_dict_name += "_clean"

        score_dict_name += f"_shallow_fusion_{settings.tokenizer}_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}_top_{settings.top_n}"
        score_file_folder = Path(settings.source_path) / score_dict_name
        score_file_folder.mkdir(parents=True, exist_ok=True)
        score_file_path = score_file_folder / f"{qid}"

        if score_file_path.with_suffix(".npz").exists():
            scores = np.load(score_file_path.with_suffix(".npz"))
            top_n_similarities = scores["top_n_similarities"]
            top_n_bm25_scores = scores["top_n_bm25_scores"]
            top_n_lcs_scores = scores["top_n_lcs_scores"]
            top_n_real_corpus_ids = scores["top_n_real_corpus_ids"]
        else:
            filtered_corpus_token = []
            # Embedding Score
            corpus_embeddings = []
            corpus_ids = []
            for i, file in enumerate(source):
                embedding_file = corpus[str(file)]["embedding_path"]
                embeddings = np.load(embedding_file)
                source_token = corpus[str(file)]["tokens"]
                assert len(source_token) == embeddings.shape[0]
                filtered_corpus_token += source_token
                corpus_embeddings.append(embeddings)
                corpus_ids += [file] * embeddings.shape[0]
            corpus_embeddings = np.vstack(corpus_embeddings)

            assert len(filtered_corpus_token) == corpus_embeddings.shape[0]

            query_embedding = np.load(query_embedding_file)
            # 計算餘弦相似度
            cosine_similarity = np.dot(query_embedding, corpus_embeddings.T) / (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(corpus_embeddings, axis=1)
            )
            assert len(filtered_corpus_token) == cosine_similarity.shape[0]

            # 取出前n個最高的餘弦相似度的索引
            top_n_indices = np.argsort(cosine_similarity)[-settings.top_n :]
            top_n_indices.sort()

            top_n_similarities = cosine_similarity[top_n_indices]

            # top n BM25
            top_n_real_corpus_ids = []
            top_n_corpus_ids = []
            aggregate_filtered_corpus = []

            now_real_corpus_id = None
            now_corpus_text = None
            now_index = -999
            for index in top_n_indices:
                tokens = filtered_corpus_token[index]
                real_corpus_id = corpus_ids[index]
                top_n_real_corpus_ids.append(real_corpus_id)

                if real_corpus_id == now_real_corpus_id:
                    if now_index + 1 == index:
                        if (
                            len(now_corpus_text) >= settings.stride
                            and len(tokens) >= settings.stride
                        ):
                            try:
                                assert (
                                    now_corpus_text[-settings.stride :]
                                    == tokens[: settings.stride]
                                )
                                now_corpus_text += tokens[settings.stride :]
                            except AssertionError:
                                now_corpus_text += tokens
                    else:
                        now_corpus_text += tokens
                else:
                    if now_real_corpus_id is not None:
                        aggregate_filtered_corpus.append(
                            self.embedding_model.tokenizer.decode(
                                now_corpus_text
                            ).replace(" ", "")
                        )
                        now_corpus_text = None
                        now_real_corpus_id = None
                    now_real_corpus_id = real_corpus_id
                    now_corpus_text = tokens

                now_index = index
                top_n_corpus_ids.append(len(aggregate_filtered_corpus))
            else:
                aggregate_filtered_corpus.append(
                    self.embedding_model.tokenizer.decode(now_corpus_text).replace(
                        " ", ""
                    )
                )
            aggregate_filtered_corpus_ws = self.ckip_ws(
                aggregate_filtered_corpus, show_progress=False
            )
            bm25 = BM25Okapi(aggregate_filtered_corpus_ws)
            tokenized_query = self.ckip_ws([query], show_progress=False)[0]
            bm25_scores = bm25.get_scores(tokenized_query)
            top_n_bm25_scores = bm25_scores[top_n_corpus_ids]

            # top n LCS
            aggregate_filtered_corpus_lcs = [
                self._lcs_length(tokenized_query, doc)
                for doc in aggregate_filtered_corpus_ws
            ]
            aggregate_filtered_corpus_lcs = np.array(aggregate_filtered_corpus_lcs)
            top_n_lcs_scores = aggregate_filtered_corpus_lcs[top_n_corpus_ids]

            np.savez(
                score_file_path,
                top_n_similarities=top_n_similarities,
                top_n_bm25_scores=top_n_bm25_scores,
                top_n_lcs_scores=top_n_lcs_scores,
                top_n_real_corpus_ids=top_n_real_corpus_ids,
            )

        # fusion
        # FIRE = 2.0  # 這個值可以根據需要調整
        # adjusted_similarities = np.exp(FIRE * top_n_similarities) - 1
        # adjusted_similarities /= np.max(adjusted_similarities)  # 正規化到 0~1

        # FIRE = 4.0  # 這個值可以根據需要調整
        # adjusted_similarities = np.where(
        #     top_n_similarities < 0.6,
        #     top_n_similarities / FIRE,
        #     top_n_similarities,  # 保持其他值不變
        # )

        top_n_fusion_scores = (
            settings.alpha * top_n_bm25_scores
            + settings.beta * top_n_lcs_scores
            + (1 - settings.alpha - settings.beta) * top_n_similarities
        )
        max_index = np.argmax(top_n_fusion_scores)
        max_id = top_n_real_corpus_ids[max_index]
        return int(max_id)

    def fusion_retrieve(self, questions, dataset):

        answer_dict_name = "answer_v10"
        if settings.clean_text:
            answer_dict_name += "_clean"

        answer_dict_name += f"_shallow_fusion_{settings.tokenizer}_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}_alpha_{settings.alpha}_beta_{settings.beta}.json"
        answer_dict_path = Path(settings.output_dir) / answer_dict_name

        answer_dict = {"answers": []}  # 初始化字典
        pbar = tqdm(total=len(questions))

        for q_dict in questions:
            if q_dict["category"] == "finance":
                # 進行檢索
                retrieved = self._fusion_retrieve(
                    q_dict["category"],
                    q_dict["qid"],
                    q_dict["query"],
                    q_dict["query_embedding"],
                    q_dict["source"],
                    dataset["finance"],
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "insurance":
                retrieved = self._fusion_retrieve(
                    q_dict["category"],
                    q_dict["qid"],
                    q_dict["query"],
                    q_dict["query_embedding"],
                    q_dict["source"],
                    dataset["insurance"],
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "faq":
                retrieved = self._fusion_retrieve(
                    q_dict["category"],
                    q_dict["qid"],
                    q_dict["query"],
                    q_dict["query_embedding"],
                    q_dict["source"],
                    dataset["faq"],
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            else:
                raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

            pbar.update(1)

        with open(answer_dict_path, "w", encoding="utf8") as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)

        return answer_dict, answer_dict_path

    def retrieve(self, questions, dataset):
        return self.fusion_retrieve(questions, dataset)
