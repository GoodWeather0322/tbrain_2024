# tbrain_2024
AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用


# 實驗結果 
## v1 version code (BM25)      
| 實驗名稱 | data_type | retriever | tokenizer | clean_text | Precision  | 備註 |
|----------|-----------|-----------|-----------|------------|------|------|
| exp 1    | pdf  | bm25      | jieba      | False       | 0.7133333  |    baseline  |
| exp 2    | ocr_text  | bm25      | jieba      | False       | 0.74  |    圖像PDF做影像辨識 |
| exp 3    | ocr_text  | bm25      | ckip      | False       | 0.7466667  |    CKIP斷詞 |
| exp 4    | ocr_text  | bm25      | ckip      | True       | 0.8  |    CKIP斷詞+文字清理 |


## v2 version code (BM25)

| 實驗名稱 | data_type | retriever | tokenizer | clean_text | Precision  | 備註 |
|----------|-----------|-----------|-----------|------------|------|------|
| exp 1    | ocr_text  | bm25      | jieba      | False       | 0.7266667  |    --  |
| exp 2    | ocr_text  | bm25      | jieba      | True       | 0.7933333  |   --  |
| exp 3    | ocr_text  | bm25      | ckip      | False       | 0.7866667  |   --  |
| exp 4    | ocr_text  | bm25      | ckip      | True       | 0.8133333  |   --  |

## v3 version code (Embedding)

| 實驗名稱 | data_type | retriever | embedding | max_tokens | stride | clean_text | Precision  | 備註 |
|----------|-----------|-----------|-----------|------------|--------|------------|------|------|
| exp 1    | ocr_text  | consine      | bge-m3      | 4096       | 3072   | True       | 0.7533333  |    --  |
| exp 2    | ocr_text  | consine      | bge-m3      | 512       | 128   | True       | 0.7933333  |    --  |
| exp 3    | ocr_text  | consine      | bge-large-zh-v1.5      | 500       | 128   | True       | 0.8066667  |    --  |
| exp 4    | ocr_text  | sparse      | bge-m3      | 4096       | 3072   | True       | 0.7933333  |    --  |

## v4 version code (Reranker)

| 實驗名稱 | data_type | reranker | max_tokens | stride | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|
| exp 1    | ocr_text  | bge-reranker-v2-m3   | 2048       | 512   | True       | 0.7933333  |    --  |

## v5 version code (BM25 + Embedding) (V2 + V3 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v3 embedding | max_tokens | stride | normalize | alpha | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3      | 4096       | 3072   | minmax   | 0.12   | True       | 0.8866667  |    alpha 0.12 最佳  |
| exp 2    | ocr_text  | ckip   | bge-m3      | 4096       | 3072   | zscore   | 0.05   | True       | 0.8733333  |    alpha 0.05最佳  |
| exp 3    | ocr_text  | ckip   | bge-m3      | 512       | 128   | minmax   | 0.12   | True       | 0.8933333  |    alpha 0.12最佳  |
| exp 3    | ocr_text  | ckip   | bge-large-zh-v1.5      | 500       | 128   | minmax   | 0.09   | True       | 0.8666667  |    alpha 0.09最佳  |
