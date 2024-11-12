from pydantic_settings import BaseSettings
from typing import Optional
import yaml


class Settings(BaseSettings):
    """應用程序配置設置類。

    這個類定義了應用程序的所有配置參數，包括路徑設置、
    模型參數和處理選項。繼承自BaseSettings以提供配置管理功能。

    Attributes:
        source_path (str): 源數據文件路徑
        question_path (str): 問題數據文件路徑
        answer_path (str): 答案數據文件路徑
        output_dir (str): 輸出目錄路徑
        data_type (str): 數據類型
        clean_text (bool): 是否清理文本
        retriever (str): 檢索器類型
        tokenizer (str): 分詞器類型
        normalize (str): 標準化方法
        retriever_v3 (str): 檢索器v3設置
        embedding_model (str): 嵌入模型名稱
        max_tokens (int): 最大標記數
        stride (int): 步長值
        alpha (float): alpha參數
        beta (float): beta參數
        top_n (int): 返回結果的數量
        scorer (bool): 是否啟用評分器
    """

    source_path: str
    question_path: str
    answer_path: str
    output_dir: str
    data_type: str
    clean_text: bool
    retriever: str
    tokenizer: str
    normalize: str
    retriever_v3: str
    embedding_model: str
    max_tokens: int
    stride: int
    alpha: float
    beta: float
    top_n: int
    scorer: bool

    class Config:
        extra = "ignore"

    @classmethod
    def from_yaml(cls, file_path: str) -> "Settings":
        """從YAML文件載入設定

        按照YAML文件中定義的設定參數創建Settings

        Args:
            file_path (str): YAML配置文件的路徑

        Returns:
            Settings: 根據YAML文件創建Settings

        """
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)


settings = Settings.from_yaml("config_v10.yaml")
