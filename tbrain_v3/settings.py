from pydantic_settings import BaseSettings
from typing import Optional
import yaml


class Settings(BaseSettings):
    source_path: str
    question_path: str
    answer_path: str
    output_dir: str
    data_type: str
    retriever: str
    embedding_model: str
    clean_text: bool
    scorer: bool

    class Config:
        extra = "ignore"

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)


settings = Settings.from_yaml("config_v3.yaml")
