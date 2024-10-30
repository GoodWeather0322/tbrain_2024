import argparse
import json
import os

from tbrain.dataloader.dataloader import DataLoader
from tbrain.retriever.retriever import Retriever
from tbrain.scorer.scorer import Scorer
from tbrain.config.settings import settings

if __name__ == "__main__":

    if settings.data_type == "pdf":
        settings.source_path = "./source/競賽資料集/reference"
        dataloader_class = DataLoader.get_dataloader("pdf")
    elif settings.data_type == "ocr_text":
        if settings.tokenizer == "jieba":
            settings.source_path = "./source/競賽資料集/reference_text"
        elif settings.tokenizer == "ckip":
            if settings.clean_text:
                settings.source_path = (
                    "./source/競賽資料集/reference_text_cleaned_ckip_converted"
                )
                settings.question_path = "./source/競賽資料集/dataset/preliminary/questions_example_cleaned_ckip.json"
            else:
                settings.source_path = (
                    "./source/競賽資料集/reference_text_ckip_converted"
                )
                settings.question_path = "./source/競賽資料集/dataset/preliminary/questions_example_ckip.json"
        dataloader_class = DataLoader.get_dataloader("ocr_text")
    else:
        raise ValueError(f"Unsupported dataloader: {settings.data_type}")

    if settings.retriever == "bm25":
        retriever_class = Retriever.get_retriever("bm25")
    else:
        raise ValueError(f"Unsupported retriever: {settings.retriever}")

    output_file_name = f"{settings.data_type}_{settings.retriever}_answers.json"
    settings.output_path = os.path.join(settings.output_dir, output_file_name)

    scorer_class = Scorer.get_scorer("precision")

    dataloader = dataloader_class(settings.source_path)
    retriever = retriever_class()
    scorer = scorer_class(settings.answer_path)

    corpus_dict = dataloader.load_data()

    answer_dict = retriever.retrieve_all(settings.question_path, corpus_dict)

    # 將答案字典保存為json文件
    with open(settings.output_path, "w", encoding="utf8") as f:
        json.dump(
            answer_dict, f, ensure_ascii=False, indent=4
        )  # 儲存檔案，確保格式和非ASCII字符

    if settings.scorer:
        scorer.score(settings.output_path)
