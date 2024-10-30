from pydantic_settings import BaseSettings
from typing import Optional
import yaml


class Settings(BaseSettings):
    question_path: str
    answer_path: str
    data_type: str
    retriever: str
    output_dir: str
    scorer: bool
    tokenizer: str
    clean_text: bool
    source_path: Optional[str] = None
    output_path: Optional[str] = None

    class Config:
        extra = "ignore"

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)


settings = Settings.from_yaml("config.yaml")
