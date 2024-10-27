import os
import json
import pdfplumber
from tqdm import tqdm


class PdfDataLoader:
    def __init__(self, source_path: str):
        self.source_path = source_path

    # 讀取單個PDF文件並返回其文本內容
    def read_pdf(self, pdf_loc, page_infos: list = None):
        pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

        # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

        # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
        pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages
        pdf_text = ""
        for _, page in enumerate(pages):  # 迴圈遍歷每一頁
            text = page.extract_text()  # 提取頁面的文本內容
            if text:
                pdf_text += text
        pdf.close()  # 關閉PDF文件

        return pdf_text  # 返回萃取出的文本

    def _load_data(self, source_path):
        masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
        corpus_dict = {
            int(file.replace(".pdf", "")): self.read_pdf(
                os.path.join(source_path, file)
            )
            for file in tqdm(masked_file_ls)
        }  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
        return corpus_dict

    def load_data(self):
        data_json_path = os.path.join(self.source_path, "data.json")
        if os.path.exists(data_json_path):
            with open(data_json_path, "r") as f:
                corpus_dict = json.load(f)
                corpus_dict_insurance = corpus_dict["insurance"]
                corpus_dict_insurance = {
                    int(key): str(value) for key, value in corpus_dict_insurance.items()
                }
                corpus_dict_finance = corpus_dict["finance"]
                corpus_dict_finance = {
                    int(key): str(value) for key, value in corpus_dict_finance.items()
                }
                corpus_dict_faq = corpus_dict["faq"]
                corpus_dict_faq = {
                    int(key): str(value) for key, value in corpus_dict_faq.items()
                }
        else:
            corpus_dict_insurance = self._load_data(
                os.path.join(self.source_path, "insurance")
            )
            corpus_dict_finance = self._load_data(
                os.path.join(self.source_path, "finance")
            )
            with open(
                os.path.join(self.source_path, "faq/pid_map_content.json"), "rb"
            ) as f_s:
                key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            corpus_dict_faq = {
                int(key): str(value) for key, value in key_to_source_dict.items()
            }
            corpus_dict = {
                "insurance": corpus_dict_insurance,
                "finance": corpus_dict_finance,
                "faq": corpus_dict_faq,
            }
            with open(data_json_path, "w") as f:
                json.dump(corpus_dict, f, ensure_ascii=False, indent=4)

        return {
            "insurance": corpus_dict_insurance,
            "finance": corpus_dict_finance,
            "faq": corpus_dict_faq,
        }
