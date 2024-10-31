"""
V2: 使用傳統方法 => 斷詞 ( jieba, ckip-transformer ) + BM25進行資料檢索
"""

from tbrain_v2.settings import settings
from tbrain_v2.dataloader import DataLoader
from tbrain_v2.retriever import Retriever
from tbrain_v2.scorer import Scorer

if __name__ == "__main__":

    dataloader = DataLoader()
    questions, related_corpus_ids = dataloader.load_dataset()
    questions, dataset = dataloader.preprocess_dataset(questions, related_corpus_ids)

    retriever = Retriever()
    answers_dict = retriever.retrieve(questions, dataset)

    if settings.scorer:
        scorer = Scorer()
        scorer.score(answers_dict)
