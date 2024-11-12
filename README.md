# tbrain_2024
AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用

## 環境

```bash
pip install -r requirements.txt
```

## 額外資料集
無

## 運行

```bash
python main.py
```

## PDF OCR 預處理
使用[DocXChain](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Applications/DocXChain)套件進行PDF OCR預處理


## 版本

- v10: 使用BM25 + Embedding + LCS方法進行資料檢索

## 參數設定

- 請參考config_v10.yaml
- alpha: 0.01
- beta: 0.15
