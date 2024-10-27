from tbrain.dataloader.pdf_dataloader import PdfDataLoader


class DataLoader:
    @classmethod
    def get_dataloader(cls, name: str):
        if name == "pdf":
            return PdfDataLoader
        else:
            raise ValueError(f"Unsupported dataloader: {name}")
