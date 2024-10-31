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
| exp 2    | ocr_text  | bm25      | jieba      | True       | 0.7733333  |   --  |
| exp 3    | ocr_text  | bm25      | ckip      | False       | 0.7866667  |   --  |
| exp 4    | ocr_text  | bm25      | ckip      | True       | 0.8066667  |   --  |

## v3 version code (Embedding)

| 實驗名稱 | data_type | retriever | embedding | clean_text | Precision  | 備註 |
|----------|-----------|-----------|-----------|------------|------|------|
| exp 1    | ocr_text  | consine      | bge-m3      | True       | 0.7533333  |    --  |