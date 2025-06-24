# OCR 基本說明

本文件以中文說明如何使用本專案進行光學字元辨識 (OCR) 任務，從下載資料集到執行步驟，並指出需要修改的檔案位置。

## 1. 環境設定

1. 建立 Python 虛擬環境並安裝依賴：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python setup.py install
   ```

2. 下載與執行模型需要較新的 `torch` 版本及 `transformers` 套件，若安裝過程有問題，請先確認 CUDA 驅動與硬體資源。

## 2. 下載資料集

專案的 OCR 範例使用 Hugging Face 上的公開資料集，例如 `iiit5k`、`svt`、`icdar2013` 或 `iam`。
執行腳本會在第一次運行時自動從 Hugging Face 下載資料集，因此僅需確保能連線至 Hugging Face Hub。

若需手動下載，可使用 `datasets` 套件：
```bash
datasets-cli download {資料集名稱}
```

下載完成後，資料集會儲存在 `~/.cache/huggingface/datasets/` 目錄下。

## 3. 執行 OCR 單張圖片辨識

範例腳本 `benchmarks/run_ocr_single_file.py` 可用於單張圖片辨識：
```bash
python benchmarks/run_ocr_single_file.py \
    --image_file_path path/to/image.png \
    --result_output_path output.json
```
預設會載入 `config_files/model/gfd-ocr-en.yaml` 設定檔。

## 4. 執行 OCR 基準測試

`benchmarks/run_ocr_benchmark.py` 可以對整個資料集進行評估：
```bash
python benchmarks/run_ocr_benchmark.py \
    --dataset_name iiit5k \
    --output_dir ocr_result/
```
可額外指定 `--image_column_name` 或 `--text_column_name` 以符合資料集欄位名稱。

## 5. 與原始 ASR 架構的差異

為了在 OCR 任務上套用 GFD，需要新增下列檔案或修改：

- `gfd/gfd_ocr.py`：實作 `BreezperOCR` 類別，將 TrOCR 模型與 Breeze LLM 進行融合。
- `benchmarks/run_ocr_single_file.py`：單張圖片推論腳本。
- `benchmarks/run_ocr_benchmark.py`：資料集評估腳本。
- `config_files/model/gfd-ocr-en.yaml`：OCR 相關的模型與裝置設定。
- `gfd/__init__.py`：匯出 `BreezperOCR` 供外部模組使用。
- `README.md`：新增 OCR 使用說明。

若要進一步修改 OCR 模型路徑或 LLM 設定，可編輯 `config_files/model/gfd-ocr-en.yaml`。

## 6. 注意事項

1. 執行 OCR 時需要相當的 GPU 記憶體（範例配置使用 TrOCR-base 與 Mistral-7B）。
2. 若環境無法連線至 Hugging Face，請先手動下載模型與資料集後設定離線路徑。
3. 套件安裝或執行時若缺少 `torch` 等依賴，請先確保系統能成功安裝對應的版本。

