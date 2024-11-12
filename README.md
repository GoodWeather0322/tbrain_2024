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
| exp 5    | adobe_ocr_text  | bm25      | jieba      | False       | 0.7266667  |    -- |
| exp 6    | adobe_ocr_text  | bm25      | jieba      | True       | 0.7866667  |    -- |
| exp 7    | adobe_ocr_text  | bm25      | ckip      | False       | 0.7866667  |    -- |
| exp 8    | adobe_ocr_text  | bm25      | ckip      | True       | 0.8  |   -- |

## v3 version code (Embedding)

| 實驗名稱 | data_type | retriever | embedding | max_tokens | stride | clean_text | Precision  | 備註 |
|----------|-----------|-----------|-----------|------------|--------|------------|------|------|
| exp 1    | ocr_text  | consine      | bge-m3      | 4096       | 3072   | True       | 0.7533333  |    --  |
| exp 2    | ocr_text  | consine      | bge-m3      | 512       | 128   | True       | 0.7933333  |    --  |
| exp 3    | ocr_text  | consine      | bge-large-zh-v1.5      | 500       | 128   | True       | 0.8066667  |    --  |
| exp 4    | ocr_text  | sparse       | bge-m3      | 4096       | 3072   | True       | 0.7933333  |    --  |
| exp 5    | ocr_text  | consine      | bge-m3      | 256       | 128   | True       | 0.86  |   --  |
| exp 6    | ocr_text  | consine      | bge-m3     | 128       | 64   | True       | 0.8333333  |   --  |
| exp 7    | ocr_text  | consine      | bge-large-zh-v1.5     | 256       | 128   | True       | 0.82  |   --  |
| exp 8    | ocr_text  | consine      | bge-large-zh-v1.5     | 128       | 64   | True       | 0.8733333  |   --  |
| exp 8    | ocr_text  | consine      | bge-m3    | 64       | 32   | True       | 0.8  |   --  |
| exp 8    | ocr_text  | consine      | bge-large-zh-v1.5     | 64       | 32   | True       | 0.8733333  |   --  |
| exp 9    | adobe_ocr_text  | consine      | bge-m3      | 4096       | 3072   | True       | 0.74  |    --  |
| exp 10   | adobe_ocr_text  | consine      | bge-m3      | 512       | 128   | True       | 0.7933333  |    --  |
| exp 11   | adobe_ocr_text  | consine      | bge-large-zh-v1.5      | 500       | 128   | True       | 0.xxx  |    --  |
| exp 12   | adobe_ocr_text  | sparse       | bge-m3      | 4096       | 3072   | True       | 0.xxx  |    --  |
| exp 13   | adobe_ocr_text  | consine      | bge-m3      | 256       | 128   | True       | 0.8533333  |   --  |
| exp 14   | adobe_ocr_text  | consine      | bge-m3     | 128       | 64   | True       | 0.8266667  |   --  |
| exp 15   | adobe_ocr_text  | consine      | bge-large-zh-v1.5     | 256       | 128   | True       | 0.8333333  |   --  |
| exp 16   | adobe_ocr_text  | consine      | bge-large-zh-v1.5     | 128       | 64   | True       | 0.8533333  |   --  |
| exp 15   | adobe_ocr_text  | consine      | bge-m3     | 64       | 32   | True       | 0.79  |   --  |
| exp 16   | adobe_ocr_text  | consine      | bge-large-zh-v1.5     | 64       | 32   | True       | 0.8533333 |   --  |

## v4 version code (Reranker)

| 實驗名稱 | data_type | reranker | max_tokens | stride | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|
| exp 1    | ocr_text  | bge-reranker-v2-m3   | 2048       | 512   | True       | 0.7933333  |    --  |
| exp 2    | ocr_text  | bge-reranker-v2-m3   | 512       | 256   | True       | 0.8733333  |    --  |
| exp 3    | ocr_text  | bge-reranker-v2-m3   | 256       | 128   | True       | 0.86  |    --  |
| exp 4    | ocr_text  | bge-reranker-v2-m3   | 128       | 64   | True       | 0.8933333  |    --  |

## v5 version code (BM25 + Embedding) (V2 + V3 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v3 embedding | max_tokens | stride | normalize | alpha | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3      | 4096       | 3072   | minmax   | 0.12   | True       | 0.8866667  |    alpha 0.12 最佳  |
| exp 2    | ocr_text  | ckip   | bge-m3      | 4096       | 3072   | zscore   | 0.05   | True       | 0.8733333  |    alpha 0.05最佳  |
| exp 3    | ocr_text  | ckip   | bge-m3      | 512       | 128   | minmax   | 0.12   | True       | 0.8933333  |    alpha 0.12最佳  |
| exp 4    | ocr_text  | ckip   | bge-large-zh-v1.5      | 500       | 128   | minmax   | 0.09   | True       | 0.8666667  |    alpha 0.09最佳  |
| exp 5    | ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.12   | True       | 0.9066666  |    alpha 0.12最佳  |
| exp 6    | ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.09   | True       | 0.9  |    alpha 0.09最佳  |
| exp 7    | ocr_text  | ckip   | bge-large-zh-v1.5    | 256       | 128   | minmax   | 0.03   | True       | 0.86  |    alpha 0.03最佳  |
| exp 8    | ocr_text  | ckip   | bge-large-zh-v1.5   | 128       | 64   | minmax   | 0.09   | True       | 0.8933333  |    alpha 0.09最佳  |

## v6 version code (BM25 + Embedding + LCS) (V2 + V3 + LCS 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v3 embedding | max_tokens | stride | normalize | alpha | beta | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3      | 4096       | 3072   | minmax   | 0.06   | 0.19   | True       | 0.9133333  |     |
| exp 2    | ocr_text  | ckip   | bge-m3      | 512       | 128   | minmax   | 0.11   | 0.02   | True       | 0.9  |     |
| exp 3    | ocr_text  | ckip   | bge-large-zh-v1.5      | 500       | 128   | minmax   | 0.03   | 0.19   | True       | 0.9  |     |
| exp 4    | ocr_text  | ckip   | bge-m3     | 256       | 128   | minmax   | 0.03   | 0.25   | True       | 0.9266667  |     |
| exp 5    | ocr_text  | ckip   | bge-m3     | 128       | 64   | minmax   | 0.04   | 0.24   | True       | 0.9133333  |     |
| exp 6    | ocr_text  | ckip   | bge-large-zh-v1.5     | 256       | 128   | minmax   | 0.02   | 0.23   | True       | 0.9  |     |
| exp 7    | ocr_text  | ckip   | bge-large-zh-v1.5     | 128       | 64   | minmax   | 0.01   | 0.24   | True       | 0.9066667  |     |

## v7 version code (chunk BM25 + Embedding) (chunk版本 V2 + V3 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v3 embedding | max_tokens | stride | normalize | alpha | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.25   | True       | 0.9133333  |  --  |
| exp 2    | ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.12   | True       | 0.8933333  |  --  |
| exp 3    | ocr_text  | ckip   | bge-large-zh-v1.5    | 256       | 128   | minmax   | 0.15  | True       | 0.9133333  |  -- |
| exp 4    | ocr_text  | ckip   | bge-large-zh-v1.5   | 128       | 64   | minmax   | 0.1  | True       | 0.92  |   --  |

## v8 version code (chunk BM25 + Embedding + LCS) (chunk版本 V2 + V3 + LCS 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v3 embedding | max_tokens | stride | normalize | alpha | beta | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.18   | 0.26   | True       | 0.9133333  |    --  |
| exp 2    | ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.09   | 0.22   | True       | 0.8933333  |   --  |
| exp 3    | ocr_text  | ckip   | bge-large-zh-v1.5    | 256       | 128   | minmax   | 0.11  | 0.24   | True       | 0.9133333  |   -- |
| exp 4    | ocr_text  | ckip   | bge-large-zh-v1.5   | 128       | 64   | minmax   | 0.07  | 0.26   | True       | 0.92  |   --  |

## v9 version code (chunk BM25 + Reranker) (chunk版本 V2 + V4 做 shallow fusion)
| 實驗名稱 | data_type | v2 tokenizer | v4 reranker | max_tokens | stride | normalize | alpha | clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|

## v10 version code (top n chunk BM25 + embedding rewrite) (先計算所有source top n cosine，在拿top n 做 BM25 & LCS,  做 shallow fusion)
| 實驗名稱 | data_type | tokenizer | embedding | max_tokens | stride | normalize | alpha | beta |  top n |  clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.01   | 0.17   | 40   | True       | 0.92  |    --  |
| exp 2    | ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.02   | 0.13   | 40   | True       | 0.9133333  |   --  |
| exp 2    | ocr_text  | ckip   | bge-m3    | 64       | 32   | minmax   | 0.01   | 0.28   | 40   | True       | 0.9133333  |   --  |
| exp 3    | ocr_text  | ckip   | bge-large-zh-v1.5    | 256       | 128   | minmax   | 0.00  | 0.14   | 40   | True       | 0.9  |   -- |
| exp 4    | ocr_text  | ckip   | bge-large-zh-v1.5   | 128       | 64   | minmax   | 0.00  | 0.29   | 40   | True       | 0.9133333  |   --  |
| exp 4    | ocr_text  | ckip   | bge-large-zh-v1.5   | 64       | 32   | minmax   | 0.03  | 0.08   | 40   | True       | 0.9333333  |   --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.01   | 0.17   | 40   | True       | 0.9066667  |    --  |
| exp 2    | adobe_ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.02   | 0.26   | 40   | True       | 0.9066667  |   --  |
| exp 2    | adobe_ocr_text  | ckip   | bge-m3    | 64       | 32   | minmax   | 0.0   | 0.29   | 40   | True       | 0.9066667  |   --  |
| exp 3    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5    | 256       | 128   | minmax   | 0.02  | 0.05   | 40   | True       | 0.8933333  |   -- |
| exp 4    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5   | 128       | 64   | minmax   | 0.0  | 0.27   | 40   | True       | 0.8866667  |   --  |
| exp 4    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5   | 64       | 32   | minmax   | 0.02  | 0.21   | 40   | True       | 0.92  |   --  |

## v11 version code (source top n chunk BM25 + embedding rewrite) (先計算個別source top n cosine，在拿top n 做 BM25 & LCS,  做 shallow fusion)
| 實驗名稱 | data_type | tokenizer | embedding | max_tokens | stride | normalize | alpha | beta |  top n |  clean_text | Precision  | 備註 |
|----------|-----------|-----------|------------|--------|------------|------|------|------|------|------|------|------|
| exp 1    | ocr_text  | ckip   | bge-m3    | 256       | 128   | minmax   | 0.01   | 0.17   | 5   | True       | 0.92  |    --  |
| exp 1    | ocr_text  | ckip   | bge-m3    | 128       | 64   | minmax   | 0.02   | 0.28   | 5   | True       | 0.91  |    --  |
| exp 1    | ocr_text  | ckip   | bge-m3    | 64       | 32   | minmax   | 0.01   | 0.12   | 5   | True       | 0.9066667  |    --  |
| exp 1    | ocr_text  | ckip   | bge-large-zh-v1.5     | 256       | 128   | minmax   | 0.01   | 0.19   | 5   | True       | 0.9066667  |    --  |
| exp 1    | ocr_text  | ckip   | bge-large-zh-v1.5     | 128       | 64   | minmax   | 0.00   | 0.29   | 5   | True       | 0.9133333  |    --  |
| exp 1    | ocr_text  | ckip   | bge-large-zh-v1.5     | 64       | 32   | minmax   | 0.01   | 0.09   | 5   | True       | 0.9266667  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-m3     | 256       | 128   | minmax   | 0.01   | 0.18   | 5   | True       | 0.9066667  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-m3     | 128       | 64   | minmax   | 0.03   | 0.07   | 5   | True       | 0.90  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-m3     | 64       | 32   | minmax   | 0.01   | 0.2   | 5   | True       | 0.8866667  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5     | 256       | 128   | minmax   | 0.00   | 0.22   | 5   | True       | 08866667  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5     | 128       | 64   | minmax   | 0.xx   | 0.xx   | 5   | True       | 0.xx  |    --  |
| exp 1    | adobe_ocr_text  | ckip   | bge-large-zh-v1.5     | 64       | 32   | minmax   | 0.xx   | 0.xx   | 5   | True       | 0.xx  |    --  |
| exp 1    | ocr_text  | ckip   | bge-large-zh-v1.5     | 64       | 32   | minmax   | 0.03   | 0.03   | 8   | True       | 0.9066667  |    --  |
