from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from FlagEmbedding import FlagReranker

from tbrain_v4.settings import settings


class Retriever:
    def __init__(self):
        if settings.reranker == "bge-reranker-v2-m3":
            self.rerank_model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

        self.rerank_score_folder = (
            Path(settings.source_path)
            / f"corpus_v4_rerank_score_{settings.reranker}_{settings.max_tokens}_{settings.stride}"
        )
        self.rerank_score_folder.mkdir(parents=True, exist_ok=True)

    def _ranker_retrieve(self, qid, query, source, corpus_dict):
        rerank_texts = []
        corpus_ids = []
        for file in source:
            corpus_chunks = corpus_dict[str(file)]
            for chunk in corpus_chunks:
                rerank_texts.append([query, chunk])
            corpus_ids += [file] * len(corpus_chunks)

        score_path = self.rerank_score_folder / f"{qid}.npy"
        if score_path.exists():
            rerank_results = np.load(score_path)
        else:
            rerank_results = self.rerank_model.compute_score(
                rerank_texts, normalize=True
            )
            np.save(score_path, rerank_results)

        max_index = np.argmax(rerank_results)
        max_id = corpus_ids[max_index]
        return max_id

    def ranker_retrieve(self, questions, dataset):

        answer_dict_name = "answer_v4"
        if settings.clean_text:
            answer_dict_name += "_clean"
        answer_dict_name += (
            f"_rerank_{settings.reranker}_{settings.max_tokens}_{settings.stride}.json"
        )
        answer_dict_path = Path(settings.output_dir) / answer_dict_name

        answer_dict = {"answers": []}  # 初始化字典
        pbar = tqdm(total=len(questions))

        for q_dict in questions:
            if q_dict["category"] == "finance":
                # 進行檢索
                retrieved = self._ranker_retrieve(
                    q_dict["qid"],
                    q_dict["query_rerank"],
                    q_dict["source"],
                    dataset["finance"],
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "insurance":
                retrieved = self._ranker_retrieve(
                    q_dict["qid"],
                    q_dict["query_rerank"],
                    q_dict["source"],
                    dataset["insurance"],
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "faq":
                retrieved = self._ranker_retrieve(
                    q_dict["qid"],
                    q_dict["query_rerank"],
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

        return answer_dict

    def retrieve(self, questions, dataset):
        return self.ranker_retrieve(questions, dataset)
