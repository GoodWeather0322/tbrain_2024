"""
V5: 使用BM25 + Embedding方法進行資料檢索
"""

from pathlib import Path
import numpy as np

from tbrain_v10.dataloader import DataLoader as DataLoaderV10
from tbrain_v10.retriever import Retriever
from tbrain_v10.scorer import Scorer
from tbrain_v10.settings import settings

if __name__ == "__main__":

    settings.output_dir = Path(settings.output_dir) / "v10_competition"
    settings.output_dir.mkdir(exist_ok=True, parents=True)

    answer_dict_path_list = []
    best_precision = 0
    best_alpha = 0
    best_beta = 0
    best_answer_dict_path = ""

    # for i in np.arange(0.00, 0.05, 0.01):
    #     for j in np.arange(0.06, 0.12, 0.01):
            # i = 0.00
            # j = 0.00
    i = 0.01
    j = 0.15
    alpha = round(i, 2)
    beta = round(j, 2)
    # if alpha + beta > 1:
    #     continue
    print(f"alpha: {alpha}, beta: {beta}")
    settings.alpha = alpha
    settings.beta = beta

    # v10
    dataloaderv10 = DataLoaderV10()
    questions, related_corpus_ids = dataloaderv10.load_dataset()
    questions, dataset = dataloaderv10.preprocess_dataset(
        questions, related_corpus_ids
    )

    retriever = Retriever()
    answers_dict, answer_dict_path = retriever.retrieve(questions, dataset)
    answer_dict_path_list.append(answer_dict_path)
    if settings.scorer:
        scorer = Scorer()
        precision = scorer.score(answers_dict)
        if precision > best_precision:
            best_precision = precision
            best_alpha = alpha
            best_beta = beta
            best_answer_dict_path = answer_dict_path

    # print(
    #     f"best_precision: {best_precision}, best_alpha: {best_alpha}, best_beta: {best_beta}"
    # )
    # for file in answer_dict_path_list:
    #     if file.name != best_answer_dict_path.name:
    #         file.unlink()
