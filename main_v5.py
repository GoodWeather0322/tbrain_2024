"""
V5: 使用BM25 + Embedding方法進行資料檢索
"""

from pathlib import Path
import numpy as np

from tbrain_v2.dataloader import DataLoader as DataLoaderV2
from tbrain_v3.dataloader import DataLoader as DataLoaderV3
from tbrain_v5.retriever import Retriever
from tbrain_v5.scorer import Scorer
from tbrain_v2.settings import settings as settings_v2
from tbrain_v3.settings import settings as settings_v3
from tbrain_v5.settings import settings

if __name__ == "__main__":

    settings.output_dir = Path(settings.output_dir) / "v5"
    settings.output_dir.mkdir(exist_ok=True, parents=True)

    answer_dict_path_list = []
    best_precision = 0
    best_alpha = 0
    best_answer_dict_path = ""

    for alpha in np.arange(0.00, 0.30, 0.01):
        alpha = round(alpha, 2)
        settings.alpha = alpha

        print(f"alpha: {alpha}")

        settings_v2.retriever = settings.retriever_v2
        settings_v2.tokenizer = settings.tokenizer
        settings_v3.retriever = settings.retriever_v3
        settings_v3.embedding_model = settings.embedding_model
        settings_v3.max_tokens = settings.max_tokens
        settings_v3.stride = settings.stride

        # v2
        dataloaderv2 = DataLoaderV2()
        questions, related_corpus_ids = dataloaderv2.load_dataset()
        questions_bm25, dataset_bm25 = dataloaderv2.preprocess_dataset(
            questions, related_corpus_ids
        )

        # v3
        dataloaderv3 = DataLoaderV3()
        questions, related_corpus_ids = dataloaderv3.load_dataset()
        questions_embedding, dataset_embedding = dataloaderv3.preprocess_dataset(
            questions, related_corpus_ids
        )

        retriever = Retriever()
        answers_dict, answer_dict_path = retriever.retrieve(
            questions_bm25, dataset_bm25, questions_embedding, dataset_embedding
        )
        answer_dict_path_list.append(answer_dict_path)
        if settings.scorer:
            scorer = Scorer()
            precision = scorer.score(answers_dict)
            if precision > best_precision:
                best_precision = precision
                best_alpha = alpha
                best_answer_dict_path = answer_dict_path

    print(f"best_precision: {best_precision}, best_alpha: {best_alpha}")
    for file in answer_dict_path_list:
        if file.name != best_answer_dict_path.name:
            file.unlink()
