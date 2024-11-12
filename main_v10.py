"""
V10: 使用BM25 + Embedding + LCS方法進行資料檢索
"""

from pathlib import Path
import numpy as np

from preprocess.dataloader import DataLoader as DataLoaderV10
from Retrieval.retriever import Retriever
from settings import settings

if __name__ == "__main__":

    settings.output_dir = Path(settings.output_dir) / "v10_competition"
    settings.output_dir.mkdir(exist_ok=True, parents=True)


    i = 0.01
    j = 0.15
    alpha = round(i, 2)
    beta = round(j, 2)

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

