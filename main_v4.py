"""
V3: 使用Embedding + RAG方法進行資料檢索
"""

from tbrain_v4.settings import settings
from tbrain_v4.dataloader import DataLoader
from tbrain_v4.retriever import Retriever
from tbrain_v4.scorer import Scorer

if __name__ == "__main__":

    dataloader = DataLoader()
    questions, related_corpus_ids = dataloader.load_dataset()
    questions, dataset = dataloader.preprocess_dataset(questions, related_corpus_ids)

    retriever = Retriever()
    answers_dict = retriever.retrieve(questions, dataset)

    if settings.scorer:
        scorer = Scorer()
        scorer.score(answers_dict)
