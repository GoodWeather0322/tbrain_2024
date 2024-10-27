import argparse
import json
import os

from tbrain.dataloader.dataloader import DataLoader
from tbrain.retriever.retriever import Retriever
from tbrain.scorer.scorer import Scorer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some paths and files.")
    parser.add_argument(
        "--question_path",
        type=str,
        required=False,
        default="./source/競賽資料集/dataset/preliminary/questions_example.json",
        help="讀取發布題目路徑",
    )  # 問題文件的路徑
    parser.add_argument(
        "--source_path",
        type=str,
        required=False,
        default="./source/競賽資料集/reference",
        help="讀取參考資料路徑",
    )  # 參考資料的路徑
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="./source/baseline/bm25_retrieve_answers.json",
        help="輸出符合參賽格式的答案路徑",
    )  # 答案輸出的路徑
    parser.add_argument(
        "--answer_path",
        type=str,
        required=False,
        default="./source/競賽資料集/dataset/preliminary/ground_truths_example.json",
        help="參考答案路徑",
    )
    parser.add_argument(
        "--scorer",
        type=bool,
        required=False,
        default=False,
        help="是否評分",
    )

    args = parser.parse_args()

    dataloader_class = DataLoader.get_dataloader("pdf")
    retriever_class = Retriever.get_retriever("bm25")
    scorer_class = Scorer.get_scorer("precision")

    dataloader = dataloader_class(args.source_path)
    retriever = retriever_class()
    scorer = scorer_class(args.answer_path)

    corpus_dict = dataloader.load_data()

    answer_dict = retriever.retrieve_all(args.question_path, corpus_dict)

    # 將答案字典保存為json文件
    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(
            answer_dict, f, ensure_ascii=False, indent=4
        )  # 儲存檔案，確保格式和非ASCII字符

    if args.scorer:
        scorer.score(args.output_path)
