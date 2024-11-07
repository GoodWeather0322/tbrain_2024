"""
V5: 使用BM25 + Embedding方法進行資料檢索
"""

from pathlib import Path
import numpy as np

from tbrain_v9.dataloader import DataLoader as DataLoaderV9
from tbrain_v4.dataloader import DataLoader as DataLoaderV4
from tbrain_v9.retriever import Retriever
from tbrain_v9.scorer import Scorer
from tbrain_v4.settings import settings as settings_v4
from tbrain_v9.settings import settings

if __name__ == "__main__":

    settings.output_dir = Path(settings.output_dir) / "v9"
    settings.output_dir.mkdir(exist_ok=True, parents=True)

    answer_dict_path_list = []
    best_precision = 0
    best_alpha = 0
    best_answer_dict_path = ""

    for alpha in np.arange(0.00, 0.50, 0.01):
        alpha = round(alpha, 2)
        settings.alpha = alpha

        print(f"alpha: {alpha}")

        settings_v4.reranker = settings.reranker
        settings_v4.max_tokens = settings.max_tokens
        settings_v4.stride = settings.stride

        # v9
        dataloaderv9 = DataLoaderV9()
        questions, related_corpus_ids = dataloaderv9.load_dataset()
        questions_bm25, dataset_bm25 = dataloaderv9.preprocess_dataset(
            questions, related_corpus_ids
        )

        # # v4
        # dataloaderv4 = DataLoaderV4()
        # questions, related_corpus_ids = dataloaderv4.load_dataset()
        # questions_embedding, dataset_embedding = dataloaderv4.preprocess_dataset(
        #     questions, related_corpus_ids
        # )

        # retriever = Retriever()
        # answers_dict, answer_dict_path = retriever.retrieve(
        #     questions_bm25, dataset_bm25, questions_embedding, dataset_embedding
        # )
        # answer_dict_path_list.append(answer_dict_path)
        # if settings.scorer:
        #     scorer = Scorer()
        #     precision = scorer.score(answers_dict)
        #     if precision > best_precision:
        #         best_precision = precision
        #         best_alpha = alpha
        #         best_answer_dict_path = answer_dict_path

        break

    # print(f"best_precision: {best_precision}, best_alpha: {best_alpha}")
    # for file in answer_dict_path_list:
    #     if file.name != best_answer_dict_path.name:
    #         file.unlink()
