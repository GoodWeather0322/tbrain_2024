from pathlib import Path
import json
import numpy as np
from tbrain_v3.settings import settings


class Retriever:
    def __init__(self):
        pass

    def _cosine_similarity_retrieve(self, query_embedding_file, source, corpus_dict):
        corpus_embeddings = []
        corpus_ids = []
        for file in source:
            embedding_file = corpus_dict[str(file)]
            embeddings = np.load(embedding_file)
            corpus_embeddings.append(embeddings)
            corpus_ids += [file] * embeddings.shape[0]
        corpus_embeddings = np.vstack(corpus_embeddings)
        assert corpus_embeddings.shape[0] == len(corpus_ids)

        query_embedding = np.load(query_embedding_file)

        # 計算餘弦相似度
        cosine_similarity = np.dot(query_embedding, corpus_embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(corpus_embeddings, axis=1)
        )

        max_index = np.argmax(cosine_similarity)
        max_id = corpus_ids[max_index]
        return max_id

    def cosine_similarity_retrieve(self, questions, dataset):

        answer_dict_name = "answer_v3"
        if settings.clean_text:
            answer_dict_name += "_clean"
        answer_dict_name += f"_embedding_{settings.embedding_model}_{settings.max_tokens}_{settings.stride}.json"
        answer_dict_path = Path(settings.output_dir) / answer_dict_name

        answer_dict = {"answers": []}  # 初始化字典

        for q_dict in questions:
            if q_dict["category"] == "finance":
                # 進行檢索
                retrieved = self._cosine_similarity_retrieve(
                    q_dict["query_embedding"], q_dict["source"], dataset["finance"]
                )
                # 將結果加入字典
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "insurance":
                retrieved = self._cosine_similarity_retrieve(
                    q_dict["query_embedding"], q_dict["source"], dataset["insurance"]
                )
                answer_dict["answers"].append(
                    {"qid": q_dict["qid"], "retrieve": retrieved}
                )

            elif q_dict["category"] == "faq":
                retrieved = self._cosine_similarity_retrieve(
                    q_dict["query_embedding"], q_dict["source"], dataset["faq"]
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
        if settings.retriever == "cosine_similarity":
            return self.cosine_similarity_retrieve(questions, dataset)
