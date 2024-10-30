from tbrain.dataloader.pdf_dataloader import PdfDataLoader
from tbrain.dataloader.ocr_text_dataloader import OcrTextDataLoader


class DataLoader:
    @classmethod
    def get_dataloader(cls, name: str):
        if name == "pdf":
            return PdfDataLoader
        elif name == "ocr_text":
            return OcrTextDataLoader
        else:
            raise ValueError(f"Unsupported dataloader: {name}")
